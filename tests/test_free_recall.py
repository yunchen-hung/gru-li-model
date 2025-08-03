import unittest
import numpy as np
from gymnasium.vector import SyncVectorEnv
from tasks.FreeRecall import FreeRecall
from tasks.wrappers import MetaLearningEnv

class TestFreeRecall(unittest.TestCase):
    def setUp(self):
        # Initialize the environment
        self.env = FreeRecall(num_features=4, feature_dim=2, sequence_len=4)

    def test_reset(self):
        # Test the reset functionality
        obs, info = self.env.reset()
        # memory_sequence = self.env.memory_sequence
        # print("memory_sequence", memory_sequence)
        # print("obs: ", obs)
        # print("info: ", info)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("phase", info)
        self.assertEqual(info["phase"], "encoding")

    def test_step(self):
        # Test the step functionality
        obs, info = self.env.reset()
        memory_sequence = self.env.memory_sequence
        memory_sequence_int = self.env._convert_item_to_int(memory_sequence)
        print("memory_sequence", memory_sequence)
        print("memory_sequence_int", memory_sequence_int)
        print("time step: ", 0)
        print("obs: ", obs)
        print("info: ", info)
        
        timestep = 1
        done = False
        while done == False:
            # action = np.zeros(self.env.action_shape)
            # action[action_int[timestep-1]] = 1
            action = self.env.action_space.sample()
            print("action: ", action)
            obs, reward, _, _, info = self.env.step(action)
            print("reward: ", reward)
            print()
            print("time step: ", timestep)
            print("info: ", info)
            print("obs: ", obs)
            done = info['done']
            timestep += 1
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("phase", info)

    def test_action_space(self):
        # Test if the action space is defined correctly
        self.assertTrue(self.env.action_space.contains(self.env.action_space.sample()))

    def test_observation_space(self):
        # Test if the observation space is defined correctly
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))


class TestMetaLearningEnv(unittest.TestCase):
    def setUp(self):
        # Initialize the environment
        # self.env = MetaLearningEnv(FreeRecall(num_features=4, feature_dim=2, sequence_len=4))
        self.env = MetaLearningEnv(FreeRecall(num_features=4, feature_dim=2, sequence_len=4))

    def test_reset(self):
        # Test the reset functionality
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("phase", info)
        self.assertEqual(info["phase"], "encoding")

    def test_step(self):
        # Test the step functionality
        obs, info = self.env.reset()
        memory_sequence = self.env.unwrapped.memory_sequence
        memory_sequence_int = self.env.unwrapped._convert_item_to_int(memory_sequence)
        print("memory_sequence", memory_sequence)
        print("memory_sequence_int", memory_sequence_int)
        print("time step: ", 0)
        print("obs: ", obs)
        print("info: ", info)
        
        timestep = 1
        done = False
        while done == False:
            # action = np.zeros(self.env.action_shape)
            # action[action_int[timestep-1]] = 1
            action = self.env.action_space.sample()
            print("action: ", action)
            obs, reward, _, _, info = self.env.step(action)
            print("reward: ", reward)
            print()
            print("time step: ", timestep)
            print("info: ", info)
            print("obs: ", obs[:16], obs[16:20], obs[20:22], obs[22:])
            done = info['done']
            timestep += 1
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("phase", info)


class TestVectorEnv(unittest.TestCase):
    def setUp(self):
        def load_single_environment():
            return MetaLearningEnv(FreeRecall(num_features=4, feature_dim=2, sequence_len=4))
        self.env = SyncVectorEnv([load_single_environment for _ in range(3)])

    def test_reset(self):
        obs, info = self.env.reset()
        print(obs)
        print(info)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("phase", info)
        self.assertEqual(info["phase"][0], "encoding")



if __name__ == '__main__':
    unittest.main()
