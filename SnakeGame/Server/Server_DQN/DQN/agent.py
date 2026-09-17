import argparse
import csv
import os
import random
from collections import deque
from statistics import mean, median
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
TARGET_UPDATE_FREQ = 1_000 # 
EPSILON_START = 1.0 # Exploração(100%)
EPSILON_MIN = 0.02 # Min de Exploração (2%)
EPSILON_DECAY_STEPS = 250_000 # Total de Passos até a exploração ir pro minimo
CHECKPOINT_EVERY_GAMES = 100  # Checkpoint

MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
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

    def load_checkpoint(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Nenhum checkpoint encontrado em: {MODEL_PATH}", flush=True)
            return False
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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

            # Formato antigo não possuía train_steps, assim volta a explorar.
            self.trainer.train_steps = checkpoint.get("train_steps", 0)
        else:
            self.trainer.update_target_model()

        self.checkpoint_loaded = True
        print(f"Checkpoint carregado: partidas={self.n_game}, recorde={self.record}, "
              f"train_steps={self.trainer.train_steps}", flush=True)
        return True

    def save_checkpoint(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({
                "checkpoint_version": 2, 
                "state_size": STATE_SIZE,
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.trainer.target_model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "n_game": self.n_game, 
                "record": self.record,
                "train_steps": self.trainer.train_steps
                }, MODEL_PATH)
        print(f"Checkpoint salvo: {MODEL_PATH}", flush=True)

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
            self.trainer.train_step(*zip(*random.sample(self.memory, BATCH_SIZE)))
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

    def train_short_memory(self, state, action, reward, next_state, done): # Treina a Rede Imediato com uma unica experiencia
        self.trainer.train_step(state, action, reward, next_state, done)

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

def train():
    agent, logger, game = Agent(), TrainingLogger(), SnakeGameAI()
    logger.sync_with_agent(agent)
    print(f"Treinamento DQN iniciado em CPU | modelo: {MODEL_PATH}", flush=True)

    try:
        while True:
            state_old = agent.get_state(game)  # Pega o estado do Jogo
            action = agent.get_action(state_old) # Agente toma uma Ação conforme o estado
            reward, done, score = game.play_step(action) # Executa a ação
            state_new = agent.get_state(game) # Pega o Novo estado
            agent.train_short_memory(state_old, action, reward, state_new, done) # Treina a rede de imediato
            agent.remember(state_old, action, reward, state_new, done) # guarda essa transição em um buffer de replay
            if done:
                game.reset()
                agent.n_game += 1
                agent.train_long_memory() # Treina com as Experiencias no Buffer
                new_record = score > agent.record

                if new_record:
                    agent.record = score
                    print(f"Novo recorde: {score}", flush=True)
                logger.log_game(score, agent.epsilon)

                if new_record or agent.n_game % CHECKPOINT_EVERY_GAMES == 0:
                    agent.save_checkpoint() # Se bateu um Record salva aquele Modelo
    except KeyboardInterrupt:
        print("\nTreinamento interrompido; salvando continuidade...", flush=True)
        agent.save_checkpoint()


def evaluate(episodes): # Avaliação da Rede Reunal 
    agent = Agent()
    if not agent.checkpoint_loaded:
        raise RuntimeError("Não há checkpoint compatível para avaliar.")
    
    agent.model.eval() # coloca a rede neural em modo de avaliação, isso desliga comportamentos que só devem existir durante o treino, como Dropout e BatchNorm calculando estatísticas de batch
    scores = []

    for _ in range(episodes): # Loop de partida
        game, done = SnakeGameAI(), False
        while not done: 
            _, done, score = game.play_step(agent.get_action(agent.get_state(game), explore=False))
        scores.append(score)
    print(f"Avaliação ({episodes} partidas, epsilon=0): melhor={max(scores)}, "
          f"média={mean(scores):.2f}, mediana={median(scores):.2f}", flush=True)


if __name__ == "__main__": # garante que esse bloco só executa quando o arquivo é rodado diretamente (não quando é importado como módulo em outro script)
    # configura um parser de argumentos de linha de comando, permitindo controlar o comportamento do script via flags no terminal
    parser = argparse.ArgumentParser(description="Treino e avaliação do DQN Snake")
    parser.add_argument("--eval", action="store_true", help="avalia sem treinar ou salvar")
    # Store_true --> é o padrão idiomático do argparse para flags booleanas(não precisa passar --eval True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    if args.eval:
        if args.episodes < 1:
            parser.error("--episodes deve ser pelo menos 1")
        evaluate(args.episodes)
    else:
        train()
"""
python snake.py         --> treina
python snake.py --eval  --> avalia com 100 episódios (padrão)
python snake.py --eval --episodes 20 --> avalia com 20 episódios
"""