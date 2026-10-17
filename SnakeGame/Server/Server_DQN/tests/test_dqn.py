"""Testes do DQN v2. Rode de Server_DQN: python -m unittest discover -s tests -v"""
import os
import random
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DQN"))
import agent as agent_mod  # noqa: E402
from agent import Agent, run_eval, train  # noqa: E402


class InTempDir(unittest.TestCase):
    # Models/ e logs/ são relativos ao cwd: nunca tocar nos modelos reais
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

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


class TestReplay(unittest.TestCase):
    def test_memory_bounded_and_training_waits_for_batch(self):
        a = Agent(load_checkpoint=False)
        s = [0.0] * agent_mod.STATE_SIZE
        for _ in range(agent_mod.BATCH_SIZE - 1):
            a.remember(s, [1, 0, 0], 0.0, s, False)
        self.assertIsNone(a.train_long_memory())
        a.remember(s, [1, 0, 0], 0.0, s, False)
        self.assertIsInstance(a.train_long_memory(), float)
        self.assertEqual(a.trainer.train_steps, 1)
        self.assertEqual(a.memory.maxlen, agent_mod.MAX_MEMORY)


class TestEval(unittest.TestCase):
    def test_eval_is_deterministic_and_restores_rng(self):
        torch.manual_seed(1)
        a = Agent(load_checkpoint=False)
        random.seed(99)
        expected_next = random.random()
        random.seed(99)
        first = run_eval(a, 3)
        self.assertEqual(random.random(), expected_next)
        self.assertEqual(first, run_eval(a, 3))


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
        torch.save(a.model.state_dict(), agent_mod.MODEL_PATH)  # formato state_dict puro
        b = Agent()
        self.assertTrue(b.checkpoint_loaded)
        self.assertEqual(b.best_eval_mean, -1.0)


class TestTrainLoop(InTempDir):
    def test_short_train_writes_best_and_never_downgrades_it(self):
        old = agent_mod.EVAL_EVERY_GAMES, agent_mod.EVAL_EPISODES
        agent_mod.EVAL_EVERY_GAMES, agent_mod.EVAL_EPISODES = 5, 2
        try:
            agent_mod.seed_everything(0)
            train(max_games=10)
            self.assertTrue(os.path.exists(agent_mod.MODEL_PATH))
            self.assertTrue(os.path.exists(agent_mod.BEST_MODEL_PATH))
            last = torch.load(agent_mod.MODEL_PATH)
            best = torch.load(agent_mod.BEST_MODEL_PATH)
            self.assertEqual(last["n_game"], 10)
            self.assertLessEqual(best["best_eval_mean"], last["best_eval_mean"])
            with open(agent_mod.EVAL_LOG_FILE, encoding="utf-8") as f:
                rows = f.read().strip().splitlines()
            self.assertEqual(len(rows), 3)  # cabeçalho + avaliações nas partidas 5 e 10
            # Retomar com um melhor alto não pode sobrescrever o best_model
            ck = torch.load(agent_mod.MODEL_PATH)
            ck["best_eval_mean"] = 1e9
            torch.save(ck, agent_mod.MODEL_PATH)
            mtime = os.path.getmtime(agent_mod.BEST_MODEL_PATH)
            train(max_games=15)
            self.assertEqual(os.path.getmtime(agent_mod.BEST_MODEL_PATH), mtime)
        finally:
            agent_mod.EVAL_EVERY_GAMES, agent_mod.EVAL_EPISODES = old


if __name__ == "__main__":
    unittest.main()
