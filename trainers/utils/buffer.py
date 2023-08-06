"""
Define a base Buffer class as well as the old ListBuffer class
and new NdArrayBuffer classes. 

NdArrayBuffer should be faster than ListBuffer and thus should be used.
"""

from abc import ABC, abstractmethod

import numpy as np

from trainers.utils.experience import Experience

class Buffer(ABC):
    """
    A base class for buffers. Can be the old ListBuffer which is likely 
    slower, or NdArrayBuffer which hopefully is faster. 
    """

    @abstractmethod
    def append_experience(self, obs : np.ndarray, act : np.ndarray, rew : float, don : bool, next_obs : np.ndarray):
        """
        Extend the buffer with a new experience consisting of the given parameters.
        :param np.ndarray obs: The new observation.
        :param np.ndarray act: The new action.
        :param float rew: The new reward.
        :param bool obs: The new done flag.
        :param np.ndarray next_obs: The new next observation.
        """
    
    @abstractmethod
    def extend_buffer(self, buffer):
        """
        Extend the buffer using the content from another buffer.
        """
    
    @abstractmethod
    def size(self):
        """
        Returns the size of the buffer.
        """
    
    @abstractmethod
    def get_components(self):
        """
        Return the observation, action, reward, done and next_observation components
        in np.ndarray.
        """

class ListBuffer(Buffer):
    """
    A ListBuffer unorderly holds experiences as a list of Experience.
    """

    def __init__(self, max_size : int, obs_shape, act_shape):
        self._list = []
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.act_shape = act_shape
    
    def append_experience(self, obs : np.ndarray, act : np.ndarray, rew : float, don : bool, next_obs : np.ndarray):
        """
        Extend the buffer with a single new experience consisting of the given parameters
        by creating one and adding it at the end of the list.
        :param np.ndarray obs: The new observation.
        :param np.ndarray act: The new action.
        :param float rew: The new reward.
        :param bool obs: The new done flag.
        :param np.ndarray next_obs: The new next observation.
        """
        new_exp = Experience(obs=obs, action=act, reward=rew, done=don, next_obs=next_obs)
        self._list.append(new_exp)
        # adjust the buffer if the list's size exceeds max_size
        self._adjust_buffer_when_full()
    
    def extend_buffer(self, buffer):
        """
        Extend the buffer using the content from another buffer.
        """
        self._list.extend(buffer._list)
        self._adjust_buffer_when_full()
    
    def _adjust_buffer_when_full(self):
        """
        Adjusts the buffer content when full.
        Right now, we remove the data at the beginning of list.
        """
        while self.size() > self.max_size: self._list.pop(0)

    def size(self):
        return len(self._list)
    
    def get_components(self):
        obs, act, rew, don, next_obs = (np.empty(shape=[self.size()] + list(self.obs_shape), dtype=np.float32), 
                                        np.empty(shape=[self.size()] + list(self.act_shape), dtype=np.float32),
                                        np.empty(shape=[self.size()],                        dtype=np.float32),
                                        np.empty(shape=[self.size()],                        dtype=np.bool8  ),
                                        np.empty(shape=[self.size()] + list(self.obs_shape), dtype=np.float32))
        for i, exp in enumerate(self._list):
            obs[i], act[i], rew[i], don[i], next_obs[i] = (exp.obs,
                                                           exp.action,
                                                           exp.reward,
                                                           exp.done,
                                                           exp.next_obs)
        return obs, act, rew, don, next_obs

class NdArrayBuffer(Buffer):
    """
    A NdArrayBuffer holds experiences as the following:

    - observations : np.ndarray of np.ndarray*
    - actions : np.ndarray of np.ndarray
    - rewards : np.ndarray of float*
    - done flags : np.ndarray of bool
    - next observations : np.ndarray of np.ndarray*
    
    *The types for observations and rewards are given as indicated in:
    https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#mlagents_envs.base_env
    """

    def __init__(self, max_buffer_size : int, obs_shape, act_shape):
        """
        Initialize a numpy buffer with max_buffer_size capacity which we control
        based on 0-based indicing with the current and max sizes.
        :param int max_buffer_size: The maximum size of the buffer. We allocate this 
        size of memory when initializing this buffer.
        :param obs_shape: The shape of the osbervation space.
        :param act_shape: The shape of the action space.
        """
        self.obs      = np.empty(shape=[max_buffer_size] + obs_shape, dtype=np.float32)
        self.actions  = np.empty(shape=[max_buffer_size] + act_shape, dtype=np.float32)
        self.rewards  = np.empty(shape=[max_buffer_size],             dtype=np.float32)
        self.dones    = np.empty(shape=[max_buffer_size],             dtype=np.float32)
        self.next_obs = np.empty(shape=[max_buffer_size] + obs_shape, dtype=np.float32)
        
        self.ptr, self.size, self.max_size = 0, 0, max_buffer_size
    
    def append_experience(self, obs : np.ndarray, act : np.ndarray, rew : float, don : bool, next_obs : np.ndarray):
        """
        Extend the buffer with a new experience consisting of the given parameters.
        :param np.ndarray obs: The new observation.
        :param np.ndarray act: The new action.
        :param float rew: The new reward.
        :param bool obs: The new done flag.
        :param np.ndarray next_obs: The new next observation.
        """
        # self.size corresponds to insert location (since we use 0-based index)
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = don
        self.next_obs[self.ptr] = next_obs
        # update size and ptr
        self.ptr = (self.ptr + 1) % self.ptr
        self.size += 1
        # if size goes beyond the maximum, we can wrap around at the beginning
        if self.size > self.max_size:
            self.size = self.max_size
    
    def extend_buffer(self, buffer):
        """
        Extend the buffer using the content from another buffer.
        """
        # self.list.extend(buffer.list)
        # self._adjust_buffer_when_full()

    def size(self):
        return self.size
    
    def get_components(self):
        pass
        # return obs, act, rew, don, next_obs

    def sample(self, num_samples : int = 1):
        pass
