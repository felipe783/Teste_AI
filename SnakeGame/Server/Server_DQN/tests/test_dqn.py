"""Testes do DQN v2. Rode de Server_DQN: python -m unittest discover -s tests -v"""
import os
import random
import signal
import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DQN"))
import agent as agent_mod  # noqa: E402
from agent import Agent, CheckpointError, Config, NStepAccumulator, RunPaths, run_eval, train  # noqa: E402
from snake_gameai import BLOCK_SIZE, Direction, Point, SnakeGameAI  # noqa: E402

ACTIONS = ([1, 0, 0], [0, 1, 0], [0, 0, 1])


def quick_config(**overrides):
    # Avaliação pequena e sem checkpoint por tempo: os testes ficam rápidos e determinísticos
    values = dict(eval_every=5, eval_episodes=2, seed=0, checkpoint_minutes=1e9, resume_warmup=256)
    values.update(overrides)
    return Config(**values)


class InTempDir(unittest.TestCase):
    # Models/ e logs/ são relativos ao cwd: nunca tocar nos modelos reais
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.paths = RunPaths()

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()


class TestTargetQ(unittest.TestCase):
    def test_terminal_target_is_reward_and_target_net_syncs(self):
        torch.manual_seed(0)
        a = Agent(load_checkpoint=False)
        trainer = a.trainer
        s = torch.rand(2, agent_mod.STATE_SIZE).numpy()
        actions = [[1, 0, 0], [0, 1, 0]]
        # Com done=True o alvo é só a recompensa: a perda de um passo não depende do próximo estado
        loss_a = Agent(load_checkpoint=False)
        loss_a.model.load_state_dict(a.model.state_dict())
        loss_a.trainer.target_model.load_state_dict(a.model.state_dict())
        l1 = trainer.train_step(s, actions, [-10.0, 10.0], s, (True, True))
        l2 = loss_a.trainer.train_step(s, actions, [-10.0, 10.0], s * 0 + 5, (True, True))
        self.assertAlmostEqual(l1, l2, places=5)

        trainer.train_steps = trainer.target_update_freq - 1
        trainer.train_step(s, actions, [0.0, 0.0], s, (False, False))
        for p, t in zip(a.model.parameters(), trainer.target_model.parameters()):
            self.assertTrue(torch.equal(p, t))

    def test_action_indices_equal_one_hot(self):
        torch.manual_seed(0)
        a, b = Agent(load_checkpoint=False), Agent(load_checkpoint=False)
        b.model.load_state_dict(a.model.state_dict())
        b.trainer.target_model.load_state_dict(a.trainer.target_model.state_dict())
        s = torch.rand(3, agent_mod.STATE_SIZE).numpy()
        l1 = a.trainer.train_step(s, [[1, 0, 0], [0, 0, 1], [0, 1, 0]], [1.0, 0.0, -1.0], s, (False, True, False))
        l2 = b.trainer.train_step(s, np.array([0, 2, 1]), [1.0, 0.0, -1.0], s, (False, True, False))
        self.assertAlmostEqual(l1, l2, places=6)

    def test_n_step_uses_gamma_to_the_n(self):
        a = Agent(load_checkpoint=False, config=Config(gamma=0.9, n_step=3))
        self.assertAlmostEqual(a.trainer.gamma, 0.9 ** 3)


class TestReplay(unittest.TestCase):
    def test_memory_bounded_and_training_waits_for_batch(self):
        a = Agent(load_checkpoint=False)
        s = [0.0] * agent_mod.STATE_SIZE
        for _ in range(a.config.batch_size - 1):
            a.remember(s, [1, 0, 0], 0.0, s, False)
        self.assertIsNone(a.train_long_memory())
        a.remember(s, [1, 0, 0], 0.0, s, False)
        self.assertIsInstance(a.train_long_memory(), float)
        self.assertEqual(a.trainer.train_steps, 1)
        self.assertEqual(a.memory.capacity, a.config.memory)

    def test_ring_buffer_overwrites_oldest_and_roundtrips(self):
        a = Agent(load_checkpoint=False, config=Config(memory=5))
        for i in range(8):
            a.remember([float(i)] * agent_mod.STATE_SIZE, i % 3, float(i), [0.0] * agent_mod.STATE_SIZE, i == 7)
        self.assertEqual(len(a.memory), 5)
        self.assertEqual(sorted(a.memory.rewards.tolist()), [3.0, 4.0, 5.0, 6.0, 7.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "replay.npz")
            a.memory.save(path)
            small = Agent(load_checkpoint=False, config=Config(memory=3))
            self.assertEqual(small.memory.load(path), 3)  # fica com as 3 mais recentes
            self.assertEqual(sorted(small.memory.rewards.tolist()), [5.0, 6.0, 7.0])
            self.assertTrue(small.memory.dones[small.memory.rewards.tolist().index(7.0)])


class TestNStep(unittest.TestCase):
    def test_one_step_is_identity(self):
        acc = NStepAccumulator(1, 0.9)
        self.assertEqual(acc.push("s0", 0, 1.0, "s1", False), [("s0", 0, 1.0, "s1", False)])
        self.assertEqual(acc.push("s1", 1, -10.0, "T", True), [("s1", 1, -10.0, "T", True)])

    def test_three_step_returns_and_flush_on_done(self):
        acc = NStepAccumulator(3, 0.5)
        self.assertEqual(acc.push("s0", 0, 1.0, "s1", False), [])
        self.assertEqual(acc.push("s1", 0, 2.0, "s2", False), [])
        self.assertEqual(acc.push("s2", 0, 4.0, "s3", False), [("s0", 0, 1 + 0.5 * 2 + 0.25 * 4, "s3", False)])
        out = acc.push("s3", 0, 8.0, "T", True)
        self.assertEqual(out, [("s1", 0, 2 + 0.5 * 4 + 0.25 * 8, "T", True),
                               ("s2", 0, 4 + 0.5 * 8, "T", True), ("s3", 0, 8.0, "T", True)])


class TestEval(unittest.TestCase):
    def test_eval_is_deterministic_and_restores_rng(self):
        torch.manual_seed(1)
        a = Agent(load_checkpoint=False)
        random.seed(99)
        expected_next = random.random()
        random.seed(99)
        first = run_eval(a, 3, 12_345)
        self.assertEqual(random.random(), expected_next)
        self.assertEqual(first, run_eval(a, 3, 12_345))


def reference_reachable(game, start, blocked):
    """BFS original (até 2026-11-05) em Points, para comparar com a versão otimizada."""
    if not game._inside_board(start) or start in blocked:
        return set()
    visited, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)):
            n = Point(x + dx, y + dy)
            if game._inside_board(n) and n not in blocked and n not in visited:
                visited.add(n)
                queue.append(n)
    return visited


def reference_analysis(game, action):
    direction = game._direction_after(game.direction, action)
    next_head = game._next_point(game.head, direction)
    if not game._inside_board(next_head) or next_head in game.snake[1:]:
        return True, 0.0, False, False
    grows = next_head == game.food
    next_snake = [next_head] + (game.snake if grows else game.snake[:-1])
    reachable = reference_reachable(game, next_head, set(next_snake[1:-1]))
    return (False, len(reachable) / game.board_cells, game.food in reachable, next_snake[-1] in reachable)


class TestEnvironment(unittest.TestCase):
    def test_fast_bfs_and_cached_space_match_reference(self):
        # Política que prefere espaço livre (partidas longas, corpo grande) com 10% de ações aleatórias
        steps = 0
        for episode in range(4):
            random.seed(episode)
            game, policy, done = SnakeGameAI(), random.Random(100 + episode), False
            while not done:
                analyses = [game.action_analysis(a) for a in ACTIONS]
                self.assertEqual(analyses, [reference_analysis(game, a) for a in ACTIONS])
                ref_space = len(reference_reachable(game, game.head, set(game.snake[1:-1]))) / game.board_cells
                self.assertEqual(game._free_space_ratio(), ref_space)
                move = policy.randrange(3) if policy.random() < 0.1 else max(
                    range(3), key=lambda i: (not analyses[i][0], analyses[i][1], policy.random()))
                old_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                reward, done, _ = game.play_step(ACTIONS[move])
                if not done and reward != 10.0:  # passo comum: recompensa recalculada pela fórmula original
                    new_space = len(reference_reachable(game, game.head, set(game.snake[1:-1]))) / game.board_cells
                    new_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                    expected = -0.01 + 0.10 * (old_distance - new_distance) / BLOCK_SIZE + 0.05 * (new_space - ref_space)
                    self.assertEqual(reward, expected)
                steps += 1
        self.assertGreater(steps, 300)

    def test_filling_the_board_ends_with_victory_not_crash(self):
        # Tabuleiro 4x2: a cobra ocupa 7 células e a comida está na última livre
        game = SnakeGameAI(w=80, h=40)
        p = lambda c, r: Point(c * BLOCK_SIZE, r * BLOCK_SIZE)  # noqa: E731
        game.snake = [p(1, 1), p(2, 1), p(3, 1), p(3, 0), p(2, 0), p(1, 0), p(0, 0)]
        game.head, game.direction, game.food = game.snake[0], Direction.LEFT, p(0, 1)
        game._invalidate()
        reward, done, score = game.play_step([1, 0, 0])
        self.assertEqual((reward, done, score, game.food), (10.0, True, 1, None))

    def test_safe_explore_never_picks_a_collision_when_avoidable(self):
        a = Agent(load_checkpoint=False, config=Config(safe_explore=True, epsilon_min=1.0, epsilon_start=1.0))
        state = np.zeros(agent_mod.STATE_SIZE, dtype=np.float32)
        state[8], state[10] = 1.0, 1.0  # reto e esquerda colidem
        random.seed(0)
        self.assertTrue(all(a.get_action(state) == [0, 1, 0] for _ in range(50)))


class TestCheckpoint(InTempDir):
    def test_roundtrip_keeps_best_eval_and_counters(self):
        a = Agent(load_checkpoint=False)
        a.n_game, a.record, a.best_eval_mean, a.trainer.train_steps = 42, 7, 3.5, 1234
        a.save_checkpoint()
        b = Agent()
        self.assertTrue(b.checkpoint_loaded)
        self.assertEqual((b.n_game, b.record, b.best_eval_mean, b.trainer.train_steps), (42, 7, 3.5, 1234))
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_old_checkpoint_without_best_eval_loads(self):
        a = Agent(load_checkpoint=False)
        os.makedirs("Models")
        torch.save(a.model.state_dict(), self.paths.model)  # formato state_dict puro
        b = Agent()
        self.assertTrue(b.checkpoint_loaded)
        self.assertEqual(b.best_eval_mean, -1.0)

    def test_save_is_atomic_and_keeps_previous_as_bak(self):
        a = Agent(load_checkpoint=False)
        a.n_game = 1
        a.save_checkpoint()
        a.n_game = 2
        a.save_checkpoint()
        self.assertEqual(torch.load(self.paths.model)["n_game"], 2)
        self.assertEqual(torch.load(self.paths.model + ".bak")["n_game"], 1)
        self.assertFalse(os.path.exists(self.paths.model + ".tmp"))

    def test_truncated_checkpoint_falls_back_to_bak(self):
        # DQN-02: antes, o arquivo truncado fazia o treino recomeçar do zero e sobrescrevê-lo
        a = Agent(load_checkpoint=False)
        a.n_game = 7
        a.save_checkpoint()
        a.save_checkpoint()  # o .bak passa a existir
        data = Path(self.paths.model).read_bytes()
        with open(self.paths.model, "wb") as f:
            f.write(data[: len(data) // 2])
        b = Agent(load_checkpoint=False)
        agent_mod.resume(b)
        self.assertEqual(b.n_game, 7)
        self.assertTrue(any(name.startswith("model.corrompido-") for name in os.listdir("Models")))

    def test_corrupt_without_bak_or_incompatible_stops_without_touching(self):
        os.makedirs("Models")
        with open(self.paths.model, "wb") as f:
            f.write(b"lixo")
        with self.assertRaises(CheckpointError):
            train(quick_config(max_games=1))
        self.assertEqual(Path(self.paths.model).read_bytes(), b"lixo")

        os.remove(self.paths.model)
        torch.save({"model_state_dict": {"linear1.weight": torch.zeros(256, 15)}}, self.paths.model)
        before = Path(self.paths.model).read_bytes()
        with self.assertRaises(CheckpointError):
            train(quick_config(max_games=1))
        self.assertEqual(Path(self.paths.model).read_bytes(), before)


class TestTrainLoop(InTempDir):
    def test_short_train_writes_best_and_never_downgrades_it(self):
        train(quick_config(max_games=10))
        self.assertTrue(os.path.exists(self.paths.model))
        self.assertTrue(os.path.exists(self.paths.best))
        last = torch.load(self.paths.model)
        best = torch.load(self.paths.best)
        self.assertEqual(last["n_game"], 10)
        self.assertLessEqual(best["best_eval_mean"], last["best_eval_mean"])
        self.assertEqual(best["eval_protocol"], {"episodes": 2, "seed": 12_345})
        with open(self.paths.eval_log, encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        self.assertEqual(len(rows), 3)  # cabeçalho + avaliações nas partidas 5 e 10
        # Um best_model com média alta (mesmo protocolo) não pode ser sobrescrito
        best["best_eval_mean"] = 1e9
        torch.save(best, self.paths.best)
        mtime = os.path.getmtime(self.paths.best)
        train(quick_config(max_games=15))
        self.assertEqual(os.path.getmtime(self.paths.best), mtime)

    def test_missing_model_does_not_overwrite_best(self):
        # DQN-01: sem model.pth, a primeira avaliação de uma rede nova sobrescrevia o best_model
        train(quick_config(max_games=5))
        best = torch.load(self.paths.best)
        best["best_eval_mean"] = 1e9
        torch.save(best, self.paths.best)
        for name in os.listdir("Models"):
            if name.startswith("model.pth"):
                os.remove(os.path.join("Models", name))
        mtime = os.path.getmtime(self.paths.best)
        train(quick_config(max_games=5))
        self.assertEqual(os.path.getmtime(self.paths.best), mtime)
        self.assertEqual(torch.load(self.paths.best)["best_eval_mean"], 1e9)

    def test_best_from_other_protocol_is_reevaluated(self):
        # Um best_model antigo (20 partidas, sem eval_protocol) tem média inflada: é reavaliado no protocolo atual
        train(quick_config(max_games=5))
        best = torch.load(self.paths.best)
        del best["eval_protocol"]
        best["best_eval_mean"] = 1e9
        torch.save(best, self.paths.best)
        agent = train(quick_config(max_games=5))
        best = torch.load(self.paths.best)
        self.assertEqual(best["eval_protocol"], {"episodes": 2, "seed": 12_345})
        self.assertLess(best["best_eval_mean"], 1e9)
        self.assertEqual(agent.best_eval_mean, best["best_eval_mean"])

    def test_resume_continues_counters_replay_and_config(self):
        first = train(quick_config(max_games=6, gamma=0.9))
        self.assertTrue(os.path.exists(self.paths.replay))
        self.assertTrue(os.path.exists(self.paths.config))
        config = agent_mod.load_config(self.paths, {"max_games": 9})
        self.assertEqual((config.gamma, config.seed, config.max_games), (0.9, 0, 9))
        second = train(config)
        self.assertEqual(second.n_game, 9)
        self.assertGreater(second.trainer.train_steps, first.trainer.train_steps)
        self.assertGreater(second.env_steps, first.env_steps)
        self.assertEqual(second.warmup, second.config.batch_size)  # buffer restaurado: sem aquecimento extra

    def test_sigterm_stops_cleanly_and_saves(self):
        # docker stop / desligamento da instância mandam SIGTERM: o treino precisa salvar antes de sair
        timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            agent = train(quick_config(eval_every=10_000))
        finally:
            timer.cancel()
        self.assertEqual(torch.load(self.paths.model)["n_game"], agent.n_game)
        self.assertTrue(os.path.exists(self.paths.replay))
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)


if __name__ == "__main__":
    unittest.main()
