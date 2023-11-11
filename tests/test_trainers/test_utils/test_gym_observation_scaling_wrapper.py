import unittest

import gymnasium
import numpy as np

from models.trainers.utils.gym_observation_scaling_wrapper import GymObservationScaling

class TestGymObservationScaling(unittest.TestCase):

    def test_high_and_low(self):
        MAX, MIN = 1.0, -1.0
        env = gymnasium.make("Pendulum-v1")
        unscaled_high, unscaled_low = env.observation_space.high, env.observation_space.low

        wrapped_env = GymObservationScaling(env, obs_min=MIN, obs_max=MAX)
        
        # ensure we are rescaling something to a different value from their originals
        self.assertFalse(np.array_equal(np.full_like(unscaled_high, MAX), unscaled_high))
        self.assertFalse(np.array_equal(np.full_like(unscaled_low,  MIN),  unscaled_low))
        # show that there is no change in action space, for example
        self.assertTrue(np.array_equal(env.action_space, wrapped_env.action_space))
        # show that the rescaled observation space matches the new one
        self.assertTrue(np.array_equal(np.full_like(unscaled_high, MAX), wrapped_env.observation_space.high))
        self.assertTrue(np.array_equal(np.full_like(unscaled_low,  MIN),  wrapped_env.observation_space.low))
        

    def test_scaling_transform(self):
        MAX, MIN = 1.0, -1.0
        env = gymnasium.make("Pendulum-v1")
        unscaled_observation, _ = env.reset(seed=123)
        unscaled_high, unscaled_low = env.observation_space.high, env.observation_space.low

        wrapped_env = GymObservationScaling(env, obs_min=MIN, obs_max=MAX)
        scaled_observation, _ = wrapped_env.reset(seed=123)

        def scaling_transform(x):
            return ((x - ((unscaled_high + unscaled_low) / 2)) / 
                    ((unscaled_high - unscaled_low) / 2) * 
                    ((MAX - MIN) / 2) + 
                    ((MAX + MIN) / 2))
        
        # ensure we are rescaling something to a different value from their originals
        self.assertFalse(np.array_equal(np.full_like(unscaled_high, MAX), unscaled_high))
        self.assertFalse(np.array_equal(np.full_like(unscaled_low, MIN),   unscaled_low))
        # check the scaling is still correct
        self.assertTrue(np.array_equal(scaled_observation, scaling_transform(unscaled_observation)))

if __name__ == "__main__":
    unittest.main()