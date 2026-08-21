import torch
import random
import numpy as np
from collections import deque
from Teste_IA.SnakeGameAI.DQN.snake_gameai import (
    SnakeGameAI,
    Direction,
    Point,
    BLOCK_SIZE
)
from model import (
    Linear_QNet,
    QTrainer,
    DEVICE
)
from Teste_IA.SnakeGameAI.DQN.Helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Quantos jogos entre cada treinamento longo
LONG_TRAIN_INTERVAL = 10

# Quantos jogos entre cada checkpoint
CHECKPOINT_INTERVAL = 100


class Agent:

    def __init__(self):

        self.n_game = 0
        self.record = 0

        # Exploração
        self.epsilon = 80

        # Desconto futuro
        self.gamma = 0.9

        # Replay memory
        self.memory = deque(maxlen=MAX_MEMORY)

        # Modelo
        self.model = Linear_QNet(
            11,
            256,
            3
        )

        # Trainer
        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

        # Carrega checkpoint
        self.n_game, self.record, self.epsilon = \
            self.model.load_checkpoint(
                self.trainer.optimizer
            )

        print(
            f'Continuando treinamento: '
            f'Game={self.n_game}, '
            f'Record={self.record}, '
            f'Epsilon={self.epsilon}'
        )

    # ==========================================================
    # ESTADO
    # ==========================================================

    def get_state(self, game):

        head = game.snake[0]

        point_l = Point(
            head.x - BLOCK_SIZE,
            head.y
        )

        point_r = Point(
            head.x + BLOCK_SIZE,
            head.y
        )

        point_u = Point(
            head.x,
            head.y - BLOCK_SIZE
        )

        point_d = Point(
            head.x,
            head.y + BLOCK_SIZE
        )

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # Danger straight

            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Danger right

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left

            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            # Direction

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food

            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(
            state,
            dtype=np.int8
        )

    # ==========================================================
    # MEMORY
    # ==========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ==========================================================
    # LONG MEMORY
    # ==========================================================

    def train_long_memory(self):

        if len(self.memory) < BATCH_SIZE:
            return

        mini_sample = random.sample(
            self.memory,
            BATCH_SIZE
        )

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    # ==========================================================
    # SHORT MEMORY
    # ==========================================================

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ==========================================================
    # ACTION
    # ==========================================================

    def get_action(self, state):

        # Diminui exploração gradualmente
        self.epsilon = max(
            5,
            80 - self.n_game
        )

        final_move = [0, 0, 0]

        # Exploração
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)

        # Exploitação
        else:

            state0 = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                prediction = self.model(state0)

            move = torch.argmax(
                prediction
            ).item()

        final_move[move] = 1

        return final_move


# ==============================================================
# TREINAMENTO
# ==============================================================

def train():

    plot_scores = []
    plot_mean_scores = []

    total_score = 0

    agent = Agent()

    game = SnakeGameAI()

    try:

        while True:

            # ==================================================
            # ESTADO ATUAL
            # ==================================================

            state_old = agent.get_state(game)

            # ==================================================
            # ESCOLHE AÇÃO
            # ==================================================

            final_move = agent.get_action(
                state_old
            )

            # ==================================================
            # EXECUTA
            # ==================================================

            reward, done, score = game.play_step(
                final_move
            )

            # ==================================================
            # NOVO ESTADO
            # ==================================================

            state_new = agent.get_state(
                game
            )

            # ==================================================
            # SHORT MEMORY
            # ==================================================

            agent.train_short_memory(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ==================================================
            # MEMORY
            # ==================================================

            agent.remember(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ==================================================
            # FIM DO JOGO
            # ==================================================

            if done:

                game.reset()

                agent.n_game += 1

                # ==============================================
                # LONG MEMORY
                # ==============================================

                if agent.n_game % LONG_TRAIN_INTERVAL == 0:

                    agent.train_long_memory()

                # ==============================================
                # NOVO RECORD
                # ==============================================

                if score > agent.record:

                    agent.record = score

                    print(
                        f'\nNovo recorde: {agent.record}'
                    )

                # ==============================================
                # VIDEO
                # ==============================================

                if score >= agent.record - 5:

                    game.save_video(
                        score,
                        agent.record
                    )

                # ==============================================
                # LOG
                # ==============================================

                print(
                    f'Game: {agent.n_game} | '
                    f'Score: {score} | '
                    f'Record: {agent.record} | '
                    f'Epsilon: {agent.epsilon}'
                )

                # ==============================================
                # CHECKPOINT
                # ==============================================

                if agent.n_game % CHECKPOINT_INTERVAL == 0:

                    agent.model.save_checkpoint(
                        agent.trainer.optimizer,
                        agent.n_game,
                        agent.record,
                        agent.epsilon
                    )

                    print(
                        'Checkpoint salvo!'
                    )

                # ==============================================
                # PLOT
                # ==============================================

                plot_scores.append(
                    score
                )

                total_score += score

                mean_score = (
                    total_score /
                    len(plot_scores)
                )

                plot_mean_scores.append(
                    mean_score
                )

                # Atualiza gráfico
                plot(
                    plot_scores,
                    plot_mean_scores
                )

    except KeyboardInterrupt:

        print(
            '\nTreinamento interrompido.'
        )

        agent.model.save_checkpoint(
            agent.trainer.optimizer,
            agent.n_game,
            agent.record,
            agent.epsilon
        )

        print(
            'Checkpoint final salvo!'
        )

        print(
            'Pode fechar o programa.'
        )


if __name__ == "__main__":

    train()