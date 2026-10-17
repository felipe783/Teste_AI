import argparse
import csv
import os
import random
from collections import deque
from statistics import mean, median, stdev
import numpy as np
import torch
from model import DEVICE, Linear_QNet, QTrainer
from snake_gameai import BLOCK_SIZE, Direction, SnakeGameAI

MAX_MEMORY = 100_000 # Tamanho Replay Buffer
"""(state, action, reward, next_state, done): eu estava no estado X, tomei a ação Y, recebi a recompensa Z, fui parar no estado W, e o jogo acabou (ou não)"""
# TODO: Isso é chamdo de Experience Replay(Paper DeepMind)
BATCH_SIZE = 256 # Qntd de Experiencias
LR = 2.5e-4 # Taxa de Aprendizado
STATE_SIZE = 21 # Qntd Estados do Jogo
HIDDEN_SIZE = 256 
GAMMA = 0.95 # Fator de Desconto
"""Decisões de curto prazo(como bater na parede/corpo) valem mais do que recompensa muito distante"""
TRAIN_EVERY_STEPS = 4 # 1 passo de gradiente (lote do replay) a cada 4 passos do ambiente
# train_steps agora conta só passos de gradiente; os dois valores abaixo foram divididos por
# TRAIN_EVERY_STEPS para manter os cronogramas antigos medidos em passos do ambiente (1 000 e 250 000)
TARGET_UPDATE_FREQ = 1_000 // TRAIN_EVERY_STEPS
EPSILON_START = 1.0 # Exploração(100%)
EPSILON_MIN = 0.02 # Min de Exploração (2%)
EPSILON_DECAY_STEPS = 250_000 // TRAIN_EVERY_STEPS # Passos de treino até a exploração ir pro minimo
CHECKPOINT_EVERY_GAMES = 100  # Checkpoint
EVAL_EVERY_GAMES = 500 # Avaliação gulosa (epsilon=0) para escolher o melhor modelo
EVAL_EPISODES = 20
EVAL_SEED = 12_345 # Sementes fixas: todo checkpoint enfrenta a mesma sequência de comidas

MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
EVAL_LOG_FILE = os.path.join(LOG_DIR, "eval_log.csv")
LOG_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024 # 5GB


class TrainingLogger:
    def __init__(self):
        self.total_games = self.record = 0
        os.makedirs(LOG_DIR, exist_ok=True)

        if os.path.exists(LOG_FILE): # Se já existir um Log anterior
            with open(LOG_FILE, "r", newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    try: # Proteger contra linhas corrompidas
                        self.total_games = max(self.total_games, int(row["partida"]))
                        self.record = max(self.record, int(row["max"]))
                    except (ValueError, KeyError):
                        pass

    def sync_with_agent(self, agent):
        self.total_games = max(self.total_games, agent.n_game) 
        self.record = max(self.record, agent.record)

    def log_game(self, score, epsilon):
        self.total_games += 1
        self.record = max(self.record, score)

        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) >= LOG_MAX_SIZE_BYTES: # Se passar de 5GB é deletado
            os.remove(LOG_FILE)
        exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file: # Abre como append pra add uma linha
            writer = csv.writer(file)
            if not exists: # Se o cabeçalho não existir(corrompeu ou foi deletado)
                writer.writerow(["partida", "score", "max", "epsilon"])
            writer.writerow([self.total_games, score, self.record, f"{epsilon:.5f}"])

        print(f"Partida: {self.total_games} | Score: {score} | Max: {self.record} | " f"Epsilon: {epsilon:.4f}", flush=True)

class Agent:
    def __init__(self, load_checkpoint=True):
        self.n_game = self.record = 0
        self.best_eval_mean = -1.0 # Melhor média de avaliação já salva em BEST_MODEL_PATH
        """Guarda tudo em um Buffer "Memory" e com isso sorteia aleatoriamente a Experiencia(Seria o Batch Size(256))"""
        self.memory = deque(maxlen=MAX_MEMORY) 
        # deque: double-ended queue, uma função do Colletions onde é otimizada para inserir e remover itens rapidamente nas duas extremidades
        # maxlen: transforma o deque numa fila circular de tamanho fixo, quando você tenta adicionar um item além da capacidade máxima, o deque automaticamente descarta o item mais antigo da outra ponta
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, 3).to(DEVICE) # 21 Entradas, 256 Neuronios, 3 Saidas(é a que é trainada)
        self.trainer = QTrainer(self.model, LR, GAMMA, TARGET_UPDATE_FREQ) # TargetNetwork 
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
        fraction = min(1.0, self.trainer.train_steps / EPSILON_DECAY_STEPS) # Vai de 0 a 1 
        return EPSILON_START + fraction * (EPSILON_MIN - EPSILON_START) # interpolação linear

    def load_checkpoint(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(f"Nenhum checkpoint encontrado em: {path}", flush=True)
            return False
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Não foi possível ler o checkpoint; iniciando rede nova: {error}", flush=True)
            return False

        # Aceita tanto o formato novo quanto um state_dict puro de versões antigas.
        # O isinstance pe uma função nativa do Python onde verifica se um objeto é de um determinado tipo
        # isinstance(checkpoint, dict) a variavel checkpoint é uma dict?
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: # model_state_dict chava da Dict
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "linear1.weight" in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = None

        if not isinstance(state_dict, dict) or "linear1.weight" not in state_dict:
            print("Checkpoint sem pesos válidos; iniciando rede nova. "
                  "O arquivo será substituído no próximo salvamento.", flush=True)
            return False

        saved_size = state_dict["linear1.weight"].shape[1]
        if saved_size != STATE_SIZE:
            print(f"Checkpoint incompatível: estado={saved_size}, esperado={STATE_SIZE}. "
                  "Iniciando uma rede nova para não misturar representações.", flush=True)
            return False
        self.model.load_state_dict(state_dict)

        if isinstance(checkpoint, dict):
            if "optimizer_state_dict" in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "target_model_state_dict" in checkpoint:
                self.trainer.load_target_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.trainer.update_target_model()

            self.n_game = checkpoint.get("n_game", 0) # Se não existir vai dar 0 pra não quebrar o Sistema
            self.record = checkpoint.get("record", 0)
            self.best_eval_mean = checkpoint.get("best_eval_mean", -1.0)

            # Formato antigo não possuía train_steps, assim volta a explorar.
            self.trainer.train_steps = checkpoint.get("train_steps", 0)
        else:
            self.trainer.update_target_model()

        self.checkpoint_loaded = True
        print(f"Checkpoint carregado: partidas={self.n_game}, recorde={self.record}, "
              f"train_steps={self.trainer.train_steps}", flush=True)
        return True

    def save_checkpoint(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
                "checkpoint_version": 2, 
                "state_size": STATE_SIZE,
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.trainer.target_model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "n_game": self.n_game, 
                "record": self.record,
                "train_steps": self.trainer.train_steps,
                "best_eval_mean": self.best_eval_mean,
                "train_every_steps": TRAIN_EVERY_STEPS # train_steps de checkpoints antigos contam 1 por passo do ambiente
                }, path)
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
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self): 
        if len(self.memory) >= BATCH_SIZE: # Treino com Replay Memory
            # np.asarray: o torch converte uma lista de arrays numpy muito devagar (aviso do próprio PyTorch)
            return self.trainer.train_step(*map(np.asarray, zip(*random.sample(self.memory, BATCH_SIZE))))
            """
            *zip --> passa cada Tupla da Lista como um argumento separada e com isso agrupa o primeiro elemento de cada argumento junto, o segundo elemento de cada argumento junto
            Antes:
            tupla1 = (state1, action1, reward1, next_state1, done1)
            tupla2 = (state2, action2, reward2, next_state2, done2)
            tupla3 = (state3, action3, reward3, next_state3, done3)
            Depois:
            (state1, state2, state3, ...)        
            (action1, action2, action3, ...)     
            (reward1, reward2, reward3, ...)     
            (next_state1, next_state2, ...)     
            (done1, done2, done3, ...)           
            E pq fazer isso?
            O train_step precisa de 5 argumentos separados cada um contendo conjuntos com 256 elementos
            """
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
        move = random.randrange(3) if explore and random.random() < self.epsilon else None  # Decose se o Movimento vai ser Aleatorio
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

def run_eval(agent, episodes, seed=EVAL_SEED):
    """Joga partidas gulosas (epsilon=0) com sementes fixas e devolve os scores.
    A partida i usa random.seed(seed + i), então modelos diferentes enfrentam as mesmas comidas
    e a comparação fica pareada. O estado do random do treino é restaurado no final."""
    rng_state = random.getstate()
    scores = []
    try:
        for i in range(episodes):
            random.seed(seed + i)
            game, done = SnakeGameAI(), False
            while not done:
                _, done, score = game.play_step(agent.get_action(agent.get_state(game), explore=False))
            scores.append(score)
    finally:
        random.setstate(rng_state)
    return scores


def log_eval(n_game, scores, best, mean_loss):
    exists = os.path.exists(EVAL_LOG_FILE)
    with open(EVAL_LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["partida", "media", "mediana", "min", "max", "melhor_media", "loss_media"])
        writer.writerow([n_game, f"{mean(scores):.3f}", median(scores), min(scores), max(scores),
                         f"{best:.3f}", "" if mean_loss is None else f"{mean_loss:.5f}"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(max_games=None):
    agent, logger, game = Agent(), TrainingLogger(), SnakeGameAI()
    logger.sync_with_agent(agent)
    print(f"Treinamento DQN iniciado em CPU | modelo: {MODEL_PATH}", flush=True)
    env_steps, losses = 0, []
    state_new = None

    try:
        while max_games is None or agent.n_game < max_games:
            # O próximo estado do passo anterior é o estado atual: evita repetir os 3 BFS do get_state
            state_old = agent.get_state(game) if state_new is None else state_new
            action = agent.get_action(state_old) # Agente toma uma Ação conforme o estado
            reward, done, score = game.play_step(action) # Executa a ação
            state_new = agent.get_state(game) # Pega o Novo estado
            agent.remember(state_old, action, reward, state_new, done) # guarda essa transição em um buffer de replay
            env_steps += 1
            if env_steps % TRAIN_EVERY_STEPS == 0:
                loss = agent.train_long_memory() # Treina com um lote aleatório do Buffer
                if loss is not None:
                    losses.append(loss)
            if done:
                game.reset()
                state_new = None # Depois do reset o estado precisa ser recalculado
                agent.n_game += 1
                new_record = score > agent.record

                if new_record:
                    agent.record = score
                    print(f"Novo recorde: {score}", flush=True)
                logger.log_game(score, agent.epsilon)

                if agent.n_game % EVAL_EVERY_GAMES == 0:
                    agent.model.eval()
                    scores = run_eval(agent, EVAL_EPISODES)
                    agent.model.train()
                    improved = mean(scores) > agent.best_eval_mean
                    if improved: # Só substitui o melhor modelo se a média gulosa melhorou
                        agent.best_eval_mean = mean(scores)
                        agent.save_checkpoint(BEST_MODEL_PATH)
                    log_eval(agent.n_game, scores, agent.best_eval_mean, mean(losses) if losses else None)
                    print(f"Avaliação partida {agent.n_game}: média={mean(scores):.2f} "
                          f"melhor_média={agent.best_eval_mean:.2f}{' (novo melhor)' if improved else ''}", flush=True)
                    losses.clear()

                # O recorde de uma partida é ruidoso; o melhor modelo é escolhido pela avaliação acima
                if new_record or agent.n_game % CHECKPOINT_EVERY_GAMES == 0:
                    agent.save_checkpoint()
        agent.save_checkpoint()
    except KeyboardInterrupt:
        print("\nTreinamento interrompido; salvando continuidade...", flush=True)
        agent.save_checkpoint()


def evaluate(episodes, model_paths, seed): # Avaliação da Rede Reunal
    all_scores = []
    for path in model_paths:
        agent = Agent(load_checkpoint=False)
        if not agent.load_checkpoint(path):
            raise RuntimeError(f"Não há checkpoint compatível para avaliar em {path}.")
        agent.model.eval() # coloca a rede neural em modo de avaliação, isso desliga comportamentos que só devem existir durante o treino, como Dropout e BatchNorm calculando estatísticas de batch
        scores = run_eval(agent, episodes, seed)
        all_scores.append(scores)
        sd = stdev(scores) if len(scores) > 1 else 0.0
        print(f"{path} (partidas de treino={agent.n_game}) | {episodes} partidas, epsilon=0, seed={seed}: "
              f"média={mean(scores):.2f} ± {1.96 * sd / len(scores) ** 0.5:.2f} (IC95%), "
              f"mediana={median(scores):.1f}, min={min(scores)}, max={max(scores)}", flush=True)

    if len(all_scores) == 2 and episodes > 1:
        # Mesmas sementes nos dois modelos: a diferença por partida remove a sorte da comida
        diffs = [b - a for a, b in zip(*all_scores)]
        half = 1.96 * stdev(diffs) / len(diffs) ** 0.5
        print(f"Diferença pareada ({model_paths[1]} - {model_paths[0]}): {mean(diffs):.2f} ± {half:.2f} (IC95%) "
              f"-> {'significativa' if abs(mean(diffs)) > half else 'não significativa'}", flush=True)


if __name__ == "__main__": # garante que esse bloco só executa quando o arquivo é rodado diretamente (não quando é importado como módulo em outro script)
    # configura um parser de argumentos de linha de comando, permitindo controlar o comportamento do script via flags no terminal
    parser = argparse.ArgumentParser(description="Treino e avaliação do DQN Snake")
    parser.add_argument("--eval", action="store_true", help="avalia sem treinar ou salvar")
    # Store_true --> é o padrão idiomático do argparse para flags booleanas(não precisa passar --eval True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", nargs="+", default=[MODEL_PATH], help="checkpoint(s) a avaliar; dois = comparação pareada")
    parser.add_argument("--seed", type=int, default=None, help="semente (treino: reprodutibilidade; avaliação: padrão EVAL_SEED)")
    parser.add_argument("--games", type=int, default=None, help="para o treino ao chegar neste total de partidas")
    args = parser.parse_args()

    if args.eval:
        if args.episodes < 1:
            parser.error("--episodes deve ser pelo menos 1")
        evaluate(args.episodes, args.model, EVAL_SEED if args.seed is None else args.seed)
    else:
        if args.seed is not None:
            seed_everything(args.seed)
        train(args.games)
"""
python snake.py         --> treina
python snake.py --eval  --> avalia com 100 episódios (padrão)
python snake.py --eval --episodes 20 --> avalia com 20 episódios
"""