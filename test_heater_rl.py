"""Run with: python -m unittest -v test_heater_rl.py"""
import json
import math
import os
import unittest

from heater_rl import (ACTIONS, STATES, HEATER_OFF, HEATER_ON, blank_q,
                       choose_action, evaluate, policy_correct_count, q_mae,
                       reward_function, theoretical_q, train, transition, update_q)


class HeaterQLearningTests(unittest.TestCase):
    def test_six_transitions_and_rewards(self):
        expected = {
            ("Cold", HEATER_ON): ("Comfortable", 10),
            ("Cold", HEATER_OFF): ("Cold", -1),
            ("Comfortable", HEATER_ON): ("Hot", -5),
            ("Comfortable", HEATER_OFF): ("Comfortable", 10),
            ("Hot", HEATER_ON): ("Hot", -5),
            ("Hot", HEATER_OFF): ("Comfortable", 10),
        }
        for (state, action), (to, reward) in expected.items():
            self.assertEqual(transition(state, action), to)
            self.assertEqual(reward_function(to), reward)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            transition("Unknown", 0)
        with self.assertRaises(ValueError):
            transition("Cold", 7)
        with self.assertRaises(ValueError):
            reward_function("Unknown")

    def test_one_bellman_update(self):
        q = blank_q()
        u = update_q("Cold", HEATER_ON, 10, "Comfortable", q, 0.1, 0.9)
        self.assertEqual(q["Cold"][HEATER_ON], 1.0)
        self.assertEqual(u["new_q"], 1.0)

    def test_defaults_and_greedy_test(self):
        out = train()
        self.assertEqual(len(out["episode_rewards"]), 500)
        self.assertEqual(len(out["cumulative_rewards"]), 500)
        self.assertEqual(policy_correct_count(out["q"]), 3)
        self.assertEqual(evaluate(out["q"])["total"], 200)
        self.assertEqual(evaluate(out["q"])["maximum"], 200)
        self.assertAlmostEqual(theoretical_q()["Comfortable"][HEATER_OFF], 100)
        self.assertAlmostEqual(theoretical_q()["Cold"][HEATER_OFF], 89)
        self.assertAlmostEqual(theoretical_q()["Hot"][HEATER_ON], 85)

    def test_reproducible_and_independent(self):
        a = train(episodes=20, seed=11)
        b = train(episodes=20, seed=11)
        c = train(episodes=20, seed=12)
        self.assertEqual(a["q"], b["q"])
        self.assertEqual(a["episode_rewards"], b["episode_rewards"])
        self.assertNotEqual(a["episode_rewards"], c["episode_rewards"])

    def test_matches_users_original_notebook_exactly(self):
        """Execute only RL cells from the original notebook and compare every output."""
        notebook_file = os.path.join(os.path.dirname(__file__), 'Zeniabe Final V.ipynb')
        if not os.path.exists(notebook_file):
            self.skipTest("Optional original notebook not bundled")
        with open(notebook_file, encoding='utf8') as f:
            cells = json.load(f)['cells']
        env = {'__name__': '__notebook_reference__'}
        for index in (1, 5, 8, 13, 22, 24, 26):
            exec(''.join(cells[index]['source']), env, env)
        our_run = train()
        self.assertEqual(env['Q'], our_run['q'])
        self.assertEqual(env['episode_rewards'], our_run['episode_rewards'])
        self.assertEqual(env['cumulative_rewards'], our_run['cumulative_rewards'])


if __name__ == '__main__':
    unittest.main()
