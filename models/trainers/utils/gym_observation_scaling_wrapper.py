"""
A wrapper that rescales observations into a given range.
"""

from typing import Union

import gymnasium
from gymnasium.spaces import Box
import numpy as np

class GymObservationScaling(gymnasium.ObservationWrapper):
    
    def __init__(self, 
                 env : gymnasium.Env, 
                 obs_min : Union[float, int, np.ndarray], 
                 obs_max : Union[float, int, np.ndarray]):
        """
        Create a wrapped environment whose maximum and minimum observation
        values are scaled to the values given.
        If any of the dimensions have max / min values of inf / -inf, 
        scaling is not well-defined so an error is raised.

        :param gymnasium.Env env: The environment we wrap.
        :param float or int or np.ndarray obs_min: The minimum values of observation 
        dimensions to scale to.
        :param float or int or np.ndarray obs_max: The maximum values of observations
         dimensions to scale to.
        """
        if np.isinf(env.observation_space.low).any() or np.isinf(env.observation_space.high).any():
            raise Exception("Some dimensions of this environment are infinite - scaling " + 
                            "will be not well-defined. Please use another wrapper!")
        assert np.less_equal(obs_min, obs_max).all(), (obs_min, obs_max)

        super().__init__(env)
        # store info from inner env
        self.inner_env_midpoint = (env.observation_space.high + env.observation_space.low) / 2
        self.inner_env_scale = env.observation_space.high - self.inner_env_midpoint
        # update observation_space to the new one
        self._observation_space = Box(shape=self.env.observation_space.shape, 
                                      low=obs_min, high=obs_max)
    
    def observation(self, obs):
        high, low = self._observation_space.high, self._observation_space.low
        neg_one_to_one = (obs - self.inner_env_midpoint) / self.inner_env_scale
        obs = (((high + low) / 2) + neg_one_to_one * (high - low) / 2)
        return np.clip(obs, a_min=low, a_max=high)
        
        
        