import unittest

import gymnasium
import numpy as np

from models.trainers.utils.gym_observation_scaling_wrapper import GymObservationScaling

class TestGymObservationScaling(unittest.TestCase):

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

        self.assertTrue(np.array_equal(scaled_observation, scaling_transform(unscaled_observation)))