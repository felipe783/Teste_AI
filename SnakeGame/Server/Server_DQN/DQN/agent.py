import argparse
import csv
import json
import math
import os
import random
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from statistics import mean, median, stdev
from typing import Optional

import numpy as np
import torch
from model import DEVICE, Linear_QNet, QTrainer
from snake_gameai import BLOCK_SIZE, Direction, SnakeGameAI

STATE_SIZE = 21 # Qntd Estados do Jogo (mudar get_state exige mudar este valor: checkpoints com outro tamanho são recusados)
HIDDEN_SIZE = 256
CHECKPOINT_VERSION = 3 # 3: config, env_steps, eval_protocol e commit no checkpoint (2026-11-05)
TERMINAL_STATE = np.zeros(STATE_SIZE, dtype=np.float32) # O próximo estado de uma transição terminal não entra no alvo (~done)
LOG_MAX_SIZE_BYTES = 256 * 1024 * 1024 # Acima disso o training_log.csv é rotacionado (renomeado), nunca apagado


@dataclass
class Config:
    """Hiperparâmetros e execução. Os padrões são os valores usados até 2026-11-05, exceto eval_episodes (20 → 50).
    Todos podem ser mudados pela linha de comando (python DQN/agent.py --help) e ficam em <run-dir>/config.json."""
    lr: float = 2.5e-4 # Taxa de Aprendizado
    gamma: float = 0.95 # Fator de Desconto
    """Decisões de curto prazo(como bater na parede/corpo) valem mais do que recompensa muito distante"""
    n_step: int = 1 # Retornos de n passos (1 = DQN padrão); o bootstrap usa gamma ** n_step
    batch_size: int = 256 # Qntd de Experiencias por passo de gradiente
    memory: int = 100_000 # Tamanho Replay Buffer
    train_every: int = 4 # 1 passo de gradiente (lote do replay) a cada 4 passos do ambiente
    # Os dois cronogramas abaixo contam passos de GRADIENTE; com train_every=4 equivalem a 1 000 e 250 000
    # passos do ambiente (os valores medidos antes de 2026-10-17)
    target_update: int = 250
    epsilon_start: float = 1.0 # Exploração(100%)
    epsilon_min: float = 0.02 # Min de Exploração (2%)
    epsilon_decay_steps: int = 62_500 # Passos de treino até a exploração ir pro minimo
    safe_explore: bool = False # Ação aleatória só entre as que não colidem no passo seguinte
    eval_every: int = 500 # Avaliação gulosa (epsilon=0) a cada N partidas para escolher o melhor modelo
    eval_episodes: int = 50 # 20 davam IC95% de ±13 pontos (DQN-03 da auditoria de 2026-11-03)
    eval_seed: int = 12_345 # Sementes da seleção: a partida i da avaliação usa eval_seed + i
    checkpoint_every: int = 100 # Checkpoint a cada N partidas...
    checkpoint_minutes: float = 10.0 # ...ou a cada N minutos, o que vier primeiro
    resume_warmup: int = 20_000 # Transições coletadas antes de treinar quando se retoma sem replay salvo
    save_replay: bool = True # Salva o replay buffer junto com o checkpoint periódico (~17 MB com 100 mil)
    seed: Optional[int] = None # None = sorteada e registrada no config.json
    threads: int = 1 # Threads do PyTorch por processo: com a rede pequena, 1 é tão rápido quanto 4 (medido)
    max_games: Optional[int] = None # Para o treino ao chegar neste total de partidas


class RunPaths:
    """Arquivos de uma execução. run_dir='.' mantém o layout antigo (Models/ e logs/ no diretório atual)."""
    def __init__(self, run_dir="."):
        self.run_dir = run_dir
        self.models = os.path.join(run_dir, "Models")
        self.model = os.path.join(self.models, "model.pth") # último estado (para retomar)
        self.best = os.path.join(self.models, "best_model.pth") # melhor média na avaliação gulosa
        self.replay = os.path.join(self.models, "replay.npz")
        self.logs = os.path.join(run_dir, "logs")
        self.train_log = os.path.join(self.logs, "training_log.csv")
        self.eval_log = os.path.join(self.logs, "eval_log.csv")
        self.config = os.path.join(run_dir, "config.json")


class CheckpointError(RuntimeError):
    """Checkpoint existente que não pode ser usado: o treino para em vez de sobrescrevê-lo."""


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def archive(path, reason):
    """Renomeia o arquivo para <nome>.<motivo>-<data><ext> (nunca apaga) e devolve o novo caminho."""
    root, ext = os.path.splitext(path)
    target = f"{root}.{reason}-{timestamp()}{ext}"
    os.replace(path, target)
    return target


def atomic_write(path, write):
    """Grava em <path>.tmp, força o disco e só então troca; a versão anterior vira <path>.bak.
    Matar o processo durante a gravação deixa no máximo um .tmp incompleto (DQN-02)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as file:
        write(file)
        file.flush()
        os.fsync(file.fileno())
    if os.path.exists(path):
        os.replace(path, f"{path}.bak")
    os.replace(tmp, path)


def read_checkpoint(path):
    """Devolve (status, checkpoint, mensagem); status: ok, missing, corrupt ou incompatible."""
    if not os.path.exists(path):
        return "missing", None, f"Nenhum checkpoint encontrado em: {path}"
    try:
        # weights_only: o checkpoint só tem tensores e tipos simples; não executa pickle arbitrário
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception as error: # arquivo truncado gera erros variados (RuntimeError, EOFError, IndexError...)
        return "corrupt", None, f"Não foi possível ler {path}: {type(error).__name__}: {error}"

    # Aceita tanto o formato novo quanto um state_dict puro de versões antigas.
    # O isinstance pe uma função nativa do Python onde verifica se um objeto é de um determinado tipo
    # isinstance(checkpoint, dict) a variavel checkpoint é uma dict?
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: # model_state_dict chava da Dict
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "linear1.weight" in checkpoint:
        state_dict, checkpoint = checkpoint, {"model_state_dict": checkpoint}
    else:
        state_dict = None
    if not isinstance(state_dict, dict) or "linear1.weight" not in state_dict:
        return "corrupt", None, f"{path} não tem pesos válidos"

    saved_size = state_dict["linear1.weight"].shape[1]
    if saved_size != STATE_SIZE:
        return "incompatible", None, (f"{path} é incompatível: estado={saved_size}, esperado={STATE_SIZE} "
                                      "(não se misturam representações)")
    return "ok", checkpoint, f"Checkpoint lido: {path}"


def git_commit():
    """Commit do código (variável GIT_COMMIT no Docker, senão git); '-dirty' se houver alteração não commitada."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "desconhecido"


class ReplayBuffer:
    """Experience Replay (Paper DeepMind) em arrays NumPy pré-alocados, usados como fila circular.
    Cada posição guarda (state, action, reward, next_state, done): eu estava no estado X, tomei a ação Y,
    recebi a recompensa Z, fui parar no estado W, e o jogo acabou (ou não).
    Antes era uma deque(maxlen=...): random.sample numa deque indexa em O(n) e cada transição era uma
    tupla de objetos; aqui o sorteio é O(lote) e a memória é ~170 bytes por transição."""

    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.pos = self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i] = next_state, done
        self.pos = (i + 1) % self.capacity # quando enche, sobrescreve a experiência mais antiga
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        index = rng.integers(0, self.size, batch_size) # com reposição; em 100 mil a diferença é desprezível
        return (self.states[index], self.actions[index], self.rewards[index],
                self.next_states[index], self.dones[index])

    def _chronological(self):
        return (self.pos - self.size + np.arange(self.size)) % self.capacity

    def save(self, path):
        order = self._chronological()
        atomic_write(path, lambda file: np.savez(
            file, states=self.states[order], actions=self.actions[order], rewards=self.rewards[order],
            next_states=self.next_states[order], dones=self.dones[order]))

    def load(self, path):
        """Carrega as transições mais recentes que couberem; devolve quantas carregou (0 se inválido)."""
        try:
            with np.load(path) as data:
                arrays = [data[key] for key in ("states", "actions", "rewards", "next_states", "dones")]
        except Exception as error:
            print(f"Replay buffer ignorado ({path}): {type(error).__name__}: {error}", flush=True)
            return 0
        states = arrays[0]
        if states.ndim != 2 or states.shape[1] != self.states.shape[1] or len({len(a) for a in arrays}) != 1:
            print(f"Replay buffer ignorado ({path}): formato incompatível", flush=True)
            return 0
        n = min(len(states), self.capacity)
        for target, source in zip((self.states, self.actions, self.rewards, self.next_states, self.dones), arrays):
            target[:n] = source[len(source) - n:]
        self.size, self.pos = n, n % self.capacity
        return n


class NStepAccumulator:
    """Transforma passos (s, a, r) em transições de n passos: R = r0 + γ r1 + ... + γ^(n-1) r(n-1).
    Com n=1 devolve exatamente a transição do passo. No fim da partida descarrega as pendentes como terminais."""

    def __init__(self, n, gamma):
        self.n, self.gamma, self.pending = n, gamma, deque()

    def push(self, state, action, reward, next_state, done):
        self.pending.append((state, action, reward))
        ready = []
        if done:
            while self.pending:
                ready.append(self._emit(next_state, True))
        elif len(self.pending) == self.n:
            ready.append(self._emit(next_state, False))
        return ready

    def _emit(self, next_state, done):
        total = sum(self.gamma ** k * reward for k, (_, _, reward) in enumerate(self.pending))
        state, action, _ = self.pending.popleft()
        return state, action, total, next_state, done


class TrainingLogger:
    def __init__(self, paths):
        self.path = paths.train_log
        self.total_games = self.record = 0
        os.makedirs(paths.logs, exist_ok=True)

        if os.path.exists(self.path): # Se já existir um Log anterior
            with open(self.path, "r", newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    try: # Proteger contra linhas corrompidas
                        self.total_games = max(self.total_games, int(row["partida"]))
                        self.record = max(self.record, int(row["max"]))
                    except (ValueError, KeyError, TypeError):
                        pass

    def sync_with_agent(self, agent):
        self.total_games = max(self.total_games, agent.n_game)
        self.record = max(self.record, agent.record)

    def log_game(self, score, epsilon):
        self.total_games += 1
        self.record = max(self.record, score)

        if os.path.exists(self.path) and os.path.getsize(self.path) >= LOG_MAX_SIZE_BYTES:
            print(f"training_log.csv rotacionado para {archive(self.path, 'parte')}", flush=True)
        exists = os.path.exists(self.path)

        with open(self.path, "a", newline="", encoding="utf-8") as file: # Abre como append pra add uma linha
            writer = csv.writer(file)
            if not exists: # Se o cabeçalho não existir(corrompeu ou foi deletado)
                writer.writerow(["partida", "score", "max", "epsilon"])
            writer.writerow([self.total_games, score, self.record, f"{epsilon:.5f}"])

        print(f"Partida: {self.total_games} | Score: {score} | Max: {self.record} | " f"Epsilon: {epsilon:.4f}", flush=True)


class Agent:
    def __init__(self, load_checkpoint=True, config=None, paths=None):
        self.config = config or Config()
        self.paths = paths or RunPaths()
        c = self.config
        self.n_game = self.record = 0
        self.env_steps = 0 # passos do ambiente acumulados entre retomadas
        self.train_seconds = 0.0 # tempo de treino acumulado entre retomadas
        self.best_eval_mean = -1.0 # Melhor média de avaliação de BEST_MODEL_PATH (-1 = nenhum ainda)
        """Guarda tudo em um Buffer "Memory" e com isso sorteia aleatoriamente a Experiencia(Seria o Batch Size(256))"""
        self.memory = ReplayBuffer(c.memory, STATE_SIZE)
        self.warmup = c.batch_size # transições mínimas no buffer antes do primeiro passo de gradiente
        self.np_rng = np.random.default_rng(c.seed)
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, 3).to(DEVICE) # 21 Entradas, 256 Neuronios, 3 Saidas(é a que é trainada)
        # Com n-step o alvo é R_n + γ^n · Q_target(s_n): o desconto do bootstrap é gamma ** n_step
        self.trainer = QTrainer(self.model, c.lr, c.gamma ** c.n_step, c.target_update) # TargetNetwork
        """
        TODO: Usar uma Estimativa Imperfeita para melhor outras Estimativa Imperfeita
        Sem a TN da o problema de "Target Problem" onde uma rede que sempre está mudando tenta prever um alvo que sempre se move
        Com a TN este "alvo" fica congelado e com isso a Rede principal pode fazer ajustes para atingir este "alvo"
        O learning rate é um parâmetro do otimizador, ele controla o tamanho dos pesos, toda vez que a Rede Erra o LR calcula a Direção que os pesos devem ir
        """
        self.checkpoint_loaded = False
        if load_checkpoint:
            self.load_checkpoint()

    @property
    # decaimento linear do Epsilon
    # Como é derivado do self.trainer.train_steps
    # Então sempre que acessar agent.epsilon o valor é recalculado
    def epsilon(self):
        c = self.config
        fraction = min(1.0, self.trainer.train_steps / c.epsilon_decay_steps) # Vai de 0 a 1
        return c.epsilon_start + fraction * (c.epsilon_min - c.epsilon_start) # interpolação linear

    def load_checkpoint(self, path=None):
        """Carrega pesos e contadores; devolve False (sem alterar nada) se o arquivo faltar ou for inválido."""
        status, checkpoint, message = read_checkpoint(path or self.paths.model)
        print(message, flush=True)
        if status != "ok":
            return False
        self.apply_checkpoint(checkpoint)
        return True

    def apply_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "target_model_state_dict" in checkpoint:
            self.trainer.load_target_state_dict(checkpoint["target_model_state_dict"])
        else:
            self.trainer.update_target_model()

        self.n_game = checkpoint.get("n_game", 0) # Se não existir vai dar 0 pra não quebrar o Sistema
        self.record = checkpoint.get("record", 0)
        self.best_eval_mean = checkpoint.get("best_eval_mean", -1.0)
        self.env_steps = checkpoint.get("env_steps", 0)
        self.train_seconds = checkpoint.get("train_seconds", 0.0)
        # Formato antigo não possuía train_steps, assim volta a explorar.
        self.trainer.train_steps = checkpoint.get("train_steps", 0)
        self.checkpoint_loaded = True
        print(f"Checkpoint carregado: partidas={self.n_game}, recorde={self.record}, "
              f"train_steps={self.trainer.train_steps}", flush=True)

    def checkpoint_dict(self):
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "state_size": STATE_SIZE,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.trainer.target_model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "n_game": self.n_game,
            "record": self.record,
            "train_steps": self.trainer.train_steps,
            "env_steps": self.env_steps,
            "train_seconds": self.train_seconds,
            "best_eval_mean": self.best_eval_mean,
            "eval_protocol": eval_protocol(self.config),
            "train_every_steps": self.config.train_every, # train_steps de checkpoints antigos contam 1 por passo do ambiente
            "config": asdict(self.config),
            "git_commit": git_commit(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save_checkpoint(self, path=None, checkpoint=None):
        path = path or self.paths.model
        data = checkpoint or self.checkpoint_dict()
        atomic_write(path, lambda file: torch.save(data, file))
        print(f"Checkpoint salvo: {path}", flush=True)

    def get_state(self, game):
        head, tail = game.head, game.snake[-1] # Posição da Cabeça e cauda

        actions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        analyses = [game.action_analysis(action) for action in actions] # Simular a posição relativa

        board_width, board_height = max(1, game.w - BLOCK_SIZE), max(1, game.h - BLOCK_SIZE) # Calcula as dimensões
        capacity = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)

        # Montar o Vetor de Decisões
        # o * antes é desempacotamento
        state = [*
            (game.direction == direction for direction in (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)),
            (game.food.x - head.x) / board_width, (game.food.y - head.y) / board_height, # Distância da comida até a cabeça
            (tail.x - head.x) / board_width, (tail.y - head.y) / board_height, # Posicação Relativa da Cauda
            *(item[0] for item in analyses),       # colisão: frente, direita, esquerda
            *(item[1] for item in analyses),       # fração de espaço acessível
            *(item[2] for item in analyses),       # rota segura até a comida
            *(item[3] for item in analyses),       # rota até a cauda/ciclo de escape
            len(game.snake) / capacity,
        ]
        return np.asarray(state, dtype=np.float32) # Converter o Lista em Python pra um NumPy, pq o Pytorch trabalha nativamente com numpy

    def remember(self, state, action, reward, next_state, done): # Pega uma Experiencia e add no Buffer
        index = action if isinstance(action, (int, np.integer)) else int(np.argmax(action)) # one-hot -> índice
        self.memory.push(state, index, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) >= max(self.warmup, self.config.batch_size): # Treino com Replay Memory
            # O sorteio já devolve 5 arrays separados (estados, ações, recompensas, próximos estados, fins),
            # cada um com batch_size linhas: é o formato que o train_step espera
            return self.trainer.train_step(*self.memory.sample(self.config.batch_size, self.np_rng))
        return None

    # train_short_memory (1 passo de gradiente com lote de 1 a cada passo do ambiente) foi removido:
    # eram ~500 atualizações online e correlacionadas por partida contra 1 do replay, e o Adam dá
    # a cada passo um tamanho parecido independentemente do lote, então o replay quase não pesava.

    def get_action(self, state, explore=True):
        # TODO: Um tensor é uma estrutura de dados que guarda números organizados em uma grade de N dimensões
        """
        Escalar	0 dimensões	5 (um número sozinho)
        Vetor	1 dimensão	[1, 2, 3]
        Matriz	2 dimensões	[[1, 2], [3, 4]]
        Tensor (genérico)	N dimensões	[[[1,2],[3,4]], [[5,6],[7,8]]] (3D, e por aí vai)

        Os tensores são eficientes por conta da sua possibilidade de serem paralelizados com a  GPU
        No codigo o Agent visualiza uim Vetor de 21 Números, quando é passado para o lote de 256 estados, isso irá virar um tensor de 2 dimesões com shape de (256,21):256 linhas (uma por experiência do batch), 21 colunas (uma por feature do estado)

        Cada camada Linear guarda 2 tensores(e todos estes tensores jutno forma o state_dict):
        self.linear1 = nn.Linear(21, 256)
        self.linear1.weight --> tensor de PESOS(uma matriz de 2 dimesões) — shape (256, 21)
        self.linear1.bias   --> tensor de BIAS(é um vetor simple 1 dimensão com 256 valores(um bias por neurônio de saída)) — shape (256,)
        TODO: Bias --> Um valor extra somado a saida de cada neurônio(um ajuste sem depender da entrada)
        TODO: Pq Weight tem entra e saida invertida? --> é só uma convenção interna da biblioteca(por causa de como a multiplicação de matrizes funciona internamente na operação de uma camada Linear: saida = entrada · Wᵀ + bias)

        A dimesão do Tensor é definada pela pergunta: quantos eixos eu preciso percorrer pra achar um número? e não "quantos conceitos diferentes existem aqui dentro?"

        tensor[x] --> 1D
        tensor[x][y] --> 2D
        tensor[x][y][z] --> 3D
        tensor[x][y]...[n] --> N

        Imagem * 2 --> está operação no Pytoch é muito otimizado e o usuario não precisa fazer um for percorrendo a imagem inteira e multiplica tudo por 2
        No pytorch ele usa Principalmente o Paralelismo entre CPU e GPU em algum lugar este for que percorre a imagem inteira é feito mas por conta do Parelilismo da CPU/GPU é feito de forma "rapida"
        Por conta de ter este parelilismo é recomandado usar um GPU por conta de ter mais unidades de Processamento e também por conta de ser feita para operações com Muitos Cálculos(matrizes)
        GPU = muitos cálculos semelhantes e independentes
        CPU = tarefas mais gerais, complexas ou com dependências sequenciais.
        """
        move = None
        if explore and random.random() < self.epsilon: # Decide se o Movimento vai ser Aleatorio
            if self.config.safe_explore:
                # state[8:11] = colisão (reto, direita, esquerda): sorteia só entre as ações que não batem
                options = [i for i in range(3) if not state[8 + i]] or [0, 1, 2]
                move = random.choice(options)
            else:
                move = random.randrange(3)
        if move is None: # Não é aleatorio
            with torch.no_grad(): # Desliga o calculo automatico de gradientes
                move = self.model(torch.as_tensor(state, dtype=torch.float32, device=DEVICE)).argmax().item()
                # Converte o array Numpy dos estados em um tensor Pytorch, e passa para a Rede, e retorna o Indice de maior valor e o .item() converte o resultado (que é um tensor PyTorch de 1 elemento) para um número Python puro (int)
                # Ex: esta linha retorna algo do tipo: [2.3, -1.1, 5.7] e o argmax() pega o maior valor neste tensor

        # Converte o Indicie em um One-hot
        # O Move vai de 0 a 2 então ele pega o Action[0,0,0] e troca o valor na posicação definada por move por 1
        # Ex: move = 2 --> action [0 , 0 , 1]
        action = [0, 0, 0]
        action[move] = 1
        return action


class EvalInterrupted(Exception):
    pass


def eval_protocol(config):
    return {"episodes": config.eval_episodes, "seed": config.eval_seed}


def ci95(scores):
    return 1.96 * stdev(scores) / len(scores) ** 0.5 if len(scores) > 1 else 0.0


def run_eval(agent, episodes, seed, should_stop=None):
    """Joga partidas gulosas (epsilon=0) com sementes fixas e devolve os scores.
    A partida i usa random.seed(seed + i): a política gulosa é determinística, então o resultado de um
    checkpoint é reprodutível, e modelos diferentes partem da mesma sequência aleatória (comparação pareada).
    As comidas só coincidem enquanto os corpos coincidem: o sorteio é entre as células livres (DQN-04).
    O estado do random do treino é restaurado no final."""
    rng_state = random.getstate()
    scores = []
    try:
        for i in range(episodes):
            random.seed(seed + i)
            game, done = SnakeGameAI(), False
            while not done:
                if should_stop is not None and should_stop():
                    raise EvalInterrupted()
                _, done, score = game.play_step(agent.get_action(agent.get_state(game), explore=False))
            scores.append(score)
    finally:
        random.setstate(rng_state)
    return scores


EVAL_HEADER = ["partida", "media", "ic95", "mediana", "min", "max", "melhor_media", "loss_media", "epsilon",
               "train_steps", "passos_ambiente", "horas_treino", "passos_por_s"]


def append_csv(path, header, row):
    """Acrescenta uma linha; se o arquivo tiver outro cabeçalho (versão antiga), ele é renomeado antes."""
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as file:
            first = next(csv.reader(file), None)
        if first != header:
            print(f"{path} tinha outro cabeçalho; arquivado em {archive(path, 'formato-antigo')}", flush=True)
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StopRequest:
    """SIGTERM (docker stop, desligamento da instância) e Ctrl+C pedem uma parada limpa: o laço termina no
    próximo passo e salva tudo. Um segundo Ctrl+C interrompe na hora."""

    def __init__(self):
        self.requested = False
        self._previous = {}

    def _handler(self, signum, frame):
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\nSinal {signal.Signals(signum).name} recebido: parando e salvando...", flush=True)

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handler)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False


def resume(agent):
    """Retoma de <run>/Models/model.pth. Arquivo ausente: começa do zero. Ilegível: usa o .bak, arquivando o
    ilegível. Sem .bak válido, ou com STATE_SIZE diferente: CheckpointError (o treino não sobrescreve nada)."""
    path = agent.paths.model
    status, checkpoint, message = read_checkpoint(path)
    if status in ("missing", "corrupt"):
        bak_status, bak, bak_message = read_checkpoint(f"{path}.bak")
        if bak_status == "ok":
            print(f"{message}\nUsando a cópia anterior: {path}.bak", flush=True)
            if status == "corrupt":
                print(f"Arquivo ilegível preservado em {archive(path, 'corrompido')}", flush=True)
            status, checkpoint = "ok", bak
        elif status == "corrupt":
            raise CheckpointError(f"{message}. A cópia {path}.bak também não serve ({bak_message}). "
                                  "Nada foi sobrescrito: mova o arquivo para começar do zero.")
    if status == "incompatible":
        raise CheckpointError(f"{message}. Nada foi sobrescrito: mova o arquivo ou use outro --run-dir.")
    if status == "missing":
        print(f"{message}. Começando do zero.", flush=True)
        return
    agent.apply_checkpoint(checkpoint)

    loaded = agent.memory.load(agent.paths.replay) if os.path.exists(agent.paths.replay) else 0
    if loaded:
        print(f"Replay buffer restaurado: {loaded} transições", flush=True)
    else:
        # Sem o buffer, os primeiros lotes viriam de poucas partidas correlacionadas (DQN-06)
        agent.warmup = max(agent.config.batch_size, min(agent.config.resume_warmup, agent.config.memory))
        print(f"Sem replay salvo: coletando {agent.warmup} transições antes de treinar", flush=True)


def protect_best(agent, should_stop=None):
    """Define agent.best_eval_mean a partir do PRÓPRIO best_model.pth, não do model.pth (DQN-01): sem isso,
    perder o model.pth fazia a primeira avaliação de uma rede nova sobrescrever o melhor modelo.
    Se o best_model foi medido com outro protocolo (nº de partidas ou sementes), ele é reavaliado no atual."""
    path = agent.paths.best
    status, checkpoint, message = read_checkpoint(path)
    if status == "missing":
        agent.best_eval_mean = -1.0
        return
    if status != "ok":
        print(f"{message}\nbest_model preservado em {archive(path, status)}", flush=True)
        agent.best_eval_mean = -1.0
        return
    protocol = eval_protocol(agent.config)
    if checkpoint.get("eval_protocol") == protocol and "best_eval_mean" in checkpoint:
        agent.best_eval_mean = checkpoint["best_eval_mean"]
    else:
        print(f"best_model.pth avaliado com outro protocolo ({checkpoint.get('eval_protocol', 'antigo: 20 partidas')}, "
              f"média gravada {checkpoint.get('best_eval_mean')}); reavaliando com {protocol}...", flush=True)
        judge = Agent(load_checkpoint=False, config=agent.config, paths=agent.paths)
        judge.model.load_state_dict(checkpoint["model_state_dict"])
        judge.model.eval()
        scores = run_eval(judge, protocol["episodes"], protocol["seed"], should_stop)
        checkpoint["best_eval_mean"], checkpoint["eval_protocol"] = mean(scores), protocol
        agent.save_checkpoint(path, checkpoint) # mesmos pesos; o anterior fica em best_model.pth.bak
        agent.best_eval_mean = checkpoint["best_eval_mean"]
    print(f"Melhor modelo atual: média {agent.best_eval_mean:.2f} ({protocol['episodes']} partidas, "
          f"sementes {protocol['seed']}+i)", flush=True)


def write_config(config, paths):
    os.makedirs(paths.run_dir or ".", exist_ok=True)
    text = json.dumps(asdict(config), indent=2, ensure_ascii=False).encode("utf-8")
    atomic_write(paths.config, lambda file: file.write(text))


def load_config(paths, overrides):
    """Padrões ← <run-dir>/config.json (retomada) ← argumentos explícitos da linha de comando."""
    config = Config()
    known = {f.name for f in fields(Config)}
    if os.path.exists(paths.config):
        with open(paths.config, encoding="utf-8") as file:
            saved = json.load(file)
        config = Config(**{k: v for k, v in saved.items() if k in known})
        changed = {k: (getattr(config, k), v) for k, v in overrides.items() if getattr(config, k) != v}
        if changed:
            print(f"Alterando a configuração salva em {paths.config}: "
                  + ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()), flush=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def train(config=None, paths=None):
    config = config or Config()
    paths = paths or RunPaths()
    if config.seed is None:
        config.seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF # sorteada, mas sempre registrada
    torch.set_num_threads(config.threads)

    with StopRequest() as stop:
        agent = Agent(load_checkpoint=False, config=config, paths=paths)
        resume(agent)
        # Semente diferente a cada retomada, mas reprodutível: seed + partidas já jogadas
        seed_everything(config.seed + agent.n_game)
        agent.np_rng = np.random.default_rng(config.seed + agent.n_game)
        write_config(config, paths)
        try:
            protect_best(agent, lambda: stop.requested)
        except EvalInterrupted:
            print("Parada pedida durante a reavaliação do best_model; nada foi alterado.", flush=True)
            return agent

        logger = TrainingLogger(paths)
        logger.sync_with_agent(agent)
        game, nstep = SnakeGameAI(), NStepAccumulator(config.n_step, config.gamma)
        print(f"Treinamento DQN iniciado em CPU ({config.threads} thread(s)) | execução: {os.path.abspath(paths.run_dir)} "
              f"| seed={config.seed} | commit={git_commit()}", flush=True)
        print("Configuração: " + json.dumps(asdict(config), ensure_ascii=False), flush=True)

        losses, recent_scores = [], deque(maxlen=100)
        state_new = None
        session_start = last_save = last_tick = time.monotonic()
        steps_at_tick = agent.env_steps

        def tick():
            nonlocal last_tick
            now = time.monotonic()
            agent.train_seconds += now - last_tick
            last_tick = now

        def save_all(with_replay):
            tick()
            agent.save_checkpoint()
            if with_replay and config.save_replay:
                agent.memory.save(paths.replay)

        try:
            while not stop.requested and (config.max_games is None or agent.n_game < config.max_games):
                # O próximo estado do passo anterior é o estado atual: evita repetir os 3 BFS do get_state
                state_old = agent.get_state(game) if state_new is None else state_new
                action = agent.get_action(state_old) # Agente toma uma Ação conforme o estado
                reward, done, score = game.play_step(action) # Executa a ação
                # Estado terminal: o alvo ignora o próximo estado (~done), então não se gastam 3 BFS nele.
                # Isso também evita o AttributeError do get_state com food=None quando a cobra enche o tabuleiro
                state_new = TERMINAL_STATE if done else agent.get_state(game)
                for transition in nstep.push(state_old, action, reward, state_new, done):
                    agent.remember(*transition) # guarda a transição no buffer de replay
                agent.env_steps += 1
                if agent.env_steps % config.train_every == 0:
                    loss = agent.train_long_memory() # Treina com um lote aleatório do Buffer
                    if loss is not None:
                        losses.append(loss)
                if not done:
                    continue

                game.reset()
                state_new = None # Depois do reset o estado precisa ser recalculado
                agent.n_game += 1
                recent_scores.append(score)
                new_record = score > agent.record
                if new_record:
                    agent.record = score
                    print(f"Novo recorde: {score}", flush=True)
                logger.log_game(score, agent.epsilon)

                if agent.n_game % config.eval_every == 0:
                    agent.model.eval()
                    try:
                        scores = run_eval(agent, config.eval_episodes, config.eval_seed, lambda: stop.requested)
                    except EvalInterrupted:
                        break
                    finally:
                        agent.model.train()
                    tick()
                    improved = mean(scores) > agent.best_eval_mean
                    if improved: # Só substitui o melhor modelo se a média gulosa melhorou
                        agent.best_eval_mean = mean(scores)
                        agent.save_checkpoint(paths.best)
                    elapsed = time.monotonic() - session_start
                    append_csv(paths.eval_log, EVAL_HEADER, [
                        agent.n_game, f"{mean(scores):.3f}", f"{ci95(scores):.3f}", median(scores), min(scores),
                        max(scores), f"{agent.best_eval_mean:.3f}", f"{mean(losses):.5f}" if losses else "",
                        f"{agent.epsilon:.5f}", agent.trainer.train_steps, agent.env_steps,
                        f"{agent.train_seconds / 3600:.3f}", f"{(agent.env_steps - steps_at_tick) / max(elapsed, 1e-9):.0f}"])
                    print(f"Avaliação partida {agent.n_game}: média={mean(scores):.2f} ± {ci95(scores):.2f} "
                          f"melhor_média={agent.best_eval_mean:.2f}{' (novo melhor)' if improved else ''}", flush=True)
                    losses.clear()

                # O recorde de uma partida é ruidoso; o melhor modelo é escolhido pela avaliação acima
                periodic = (agent.n_game % config.checkpoint_every == 0
                            or time.monotonic() - last_save >= config.checkpoint_minutes * 60)
                if new_record or periodic:
                    save_all(with_replay=periodic)
                if periodic:
                    last_save = time.monotonic()
                    elapsed = time.monotonic() - session_start
                    print(f"[status] partidas={agent.n_game} média_treino(100)={mean(recent_scores):.1f} "
                          f"ε={agent.epsilon:.3f} passos/s={(agent.env_steps - steps_at_tick) / max(elapsed, 1e-9):.0f} "
                          f"horas_treino={agent.train_seconds / 3600:.2f} replay={len(agent.memory)}", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupção forçada; salvando continuidade...", flush=True)
        save_all(with_replay=True)
    return agent


def evaluate(episodes, model_paths, seed): # Avaliação da Rede Reunal
    all_scores = []
    for path in model_paths:
        agent = Agent(load_checkpoint=False)
        if not agent.load_checkpoint(path):
            raise RuntimeError(f"Não há checkpoint compatível para avaliar em {path}.")
        agent.model.eval() # coloca a rede neural em modo de avaliação, isso desliga comportamentos que só devem existir durante o treino, como Dropout e BatchNorm calculando estatísticas de batch
        scores = run_eval(agent, episodes, seed)
        all_scores.append(scores)
        print(f"{path} (partidas de treino={agent.n_game}) | {episodes} partidas, epsilon=0, seed={seed}: "
              f"média={mean(scores):.2f} ± {ci95(scores):.2f} (IC95%), "
              f"mediana={median(scores):.1f}, min={min(scores)}, max={max(scores)}", flush=True)

    if len(all_scores) == 2 and episodes > 1:
        # Mesmas sementes nos dois modelos (números aleatórios comuns): a diferença por partida reduz a
        # variância, mas as comidas só coincidem até os corpos divergirem (DQN-04)
        diffs = [b - a for a, b in zip(*all_scores)]
        half = ci95(diffs)
        print(f"Diferença pareada ({model_paths[1]} - {model_paths[0]}): {mean(diffs):.2f} ± {half:.2f} (IC95%) "
              f"-> {'significativa' if abs(mean(diffs)) > half else 'não significativa'}", flush=True)
    return all_scores


# Argumentos de treino -> campo do Config. O padrão None indica "não informado": vale o config.json ou o padrão
TRAIN_ARGS = {
    "--lr": ("lr", float), "--gamma": ("gamma", float), "--n-step": ("n_step", int),
    "--batch-size": ("batch_size", int), "--memory": ("memory", int), "--train-every": ("train_every", int),
    "--target-update": ("target_update", int), "--epsilon-min": ("epsilon_min", float),
    "--epsilon-decay-steps": ("epsilon_decay_steps", int), "--eval-every": ("eval_every", int),
    "--eval-episodes": ("eval_episodes", int), "--checkpoint-every": ("checkpoint_every", int),
    "--checkpoint-minutes": ("checkpoint_minutes", float), "--resume-warmup": ("resume_warmup", int),
    "--threads": ("threads", int), "--games": ("max_games", int),
}


def main(argv=None):
    # configura um parser de argumentos de linha de comando, permitindo controlar o comportamento do script via flags no terminal
    parser = argparse.ArgumentParser(description="Treino e avaliação do DQN Snake. Argumentos de treino não "
                                     "informados vêm do <run-dir>/config.json (retomada) ou do padrão.")
    parser.add_argument("--eval", action="store_true", help="avalia sem treinar ou salvar")
    # Store_true --> é o padrão idiomático do argparse para flags booleanas(não precisa passar --eval True)
    parser.add_argument("--episodes", type=int, default=100, help="partidas da avaliação (--eval); padrão 100")
    parser.add_argument("--model", nargs="+", default=None, help="checkpoint(s) a avaliar; dois = comparação pareada "
                        "(padrão: <run-dir>/Models/model.pth)")
    parser.add_argument("--seed", type=int, default=None, help="semente (treino: reprodutibilidade; avaliação: padrão 12345)")
    parser.add_argument("--run-dir", default=".", help="pasta da execução: Models/, logs/ e config.json ficam nela; padrão .")
    for flag, (name, kind) in TRAIN_ARGS.items():
        parser.add_argument(flag, type=kind, default=None, dest=name, help=f"padrão {getattr(Config(), name)}")
    parser.add_argument("--safe-explore", action=argparse.BooleanOptionalAction, default=None, dest="safe_explore",
                        help="exploração só entre ações que não colidem (padrão: desligado)")
    parser.add_argument("--save-replay", action=argparse.BooleanOptionalAction, default=None, dest="save_replay",
                        help="salvar o replay buffer nos checkpoints periódicos (padrão: ligado)")
    args = parser.parse_args(argv)
    paths = RunPaths(args.run_dir)

    if args.eval:
        if args.episodes < 1:
            parser.error("--episodes deve ser pelo menos 1")
        evaluate(args.episodes, args.model or [paths.model], Config().eval_seed if args.seed is None else args.seed)
        return 0

    names = [name for name, _ in TRAIN_ARGS.values()] + ["safe_explore", "save_replay"]
    overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = load_config(paths, overrides)
    for name in ("batch_size", "memory", "train_every", "target_update", "epsilon_decay_steps", "eval_every",
                 "eval_episodes", "checkpoint_every", "threads", "n_step"):
        if getattr(config, name) < 1:
            parser.error(f"{name} deve ser pelo menos 1")
    if not 0 < config.gamma <= 1 or not math.isfinite(config.lr) or config.lr <= 0:
        parser.error("gamma deve estar em (0, 1] e lr deve ser positivo")
    try:
        train(config, paths)
    except CheckpointError as error:
        print(f"ERRO: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__": # garante que esse bloco só executa quando o arquivo é rodado diretamente (não quando é importado como módulo em outro script)
    raise SystemExit(main())
"""
python DQN/agent.py                                   --> treina (ou retoma) em ./Models e ./logs
python DQN/agent.py --run-dir runs/gamma099 --gamma 0.99 --seed 1 --games 20000 --> experimento isolado
python DQN/agent.py --eval                            --> avalia com 100 episódios (padrão)
python DQN/agent.py --eval --episodes 20              --> avalia com 20 episódios
"""
