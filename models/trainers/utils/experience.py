"""
Defining an Experience as NamedTuple, and a Trajectory as ordered list of Experiences
"""

from typing import List, NamedTuple

import numpy as np

class Experience(NamedTuple):
    """
    A single transition experience in the given world, consisting of:
    - observation : an np.ndarray*
    - action : np.ndarray
    - reward : float*
    - done flag : bool
    - next observation : an np.ndarray*
    
    *The types for observations and rewards are given as indicated in:
    https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#mlagents_envs.base_env
    """
    obs : np.ndarray
    action : np.ndarray
    reward : float
    done : bool
    next_obs : np.ndarray

    def is_equal(self, obj):
        return (isinstance(obj, Experience) and 
                np.array_equal(obj.obs, self.obs) and
                np.array_equal(obj.action, self.action) and
                obj.reward == self.reward and
                obj.done == self.done and
                np.array_equal(obj.next_obs, self.next_obs))

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]