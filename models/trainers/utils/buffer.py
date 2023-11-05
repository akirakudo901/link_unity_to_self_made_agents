"""
Define a base Buffer class as well as the old ListBuffer class
and new NdArrayBuffer classes. 

NdArrayBuffer should be faster than ListBuffer and thus should be used.
"""

from abc import ABC, abstractmethod
import os
import random
from typing import List, Tuple

import numpy as np

from models.trainers.utils.experience import Experience

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
    def sample_random_experiences(self, num_samples : int, seed : int=None):
        """
        Returns num_samples random experiences taken from the buffer, returning
        numpy arrays for corresponding components.
        Takes in a seed for reproducibility.
        :param int num_samples: The number of samples we pick. The returned number of \
            samples is the minimum of this and the size of the buffer.
        :param int seed: The seed value useful for reproducible sampling. Not applied if None.
        """
    
    @abstractmethod
    def get_components(self):
        """
        Return the observation, action, reward, done and next_observation components
        in np.ndarray.
        """
    
    @abstractmethod
    def shuffle(self):
        """
        Shuffle the experiences held in this buffer such that accessing the first
        N experiences becomes random sampling.
        """

    @abstractmethod
    def save(self, save_dir : str, file_name : str):
        """
        Saves the experiences held in this buffer to the given saving
        directory with the given file name.        

        :param str save_dir: The directory to which we save the buffer data.
        :param str file_name: The name of the saved file.
        """
        pass

    @abstractmethod
    def load(self, path : str):
        """
        Loads the experiences held in a file located at the given path.

        :param str path: The path to the file holding experience data.
        """
        pass

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
        Result will depend on the implementation of _adjust_buffer_when_full.
        """
        self._list.extend(buffer._list)
        self._adjust_buffer_when_full()
    
    def _adjust_buffer_when_full(self):
        """
        Adjusts the buffer content when full.
        Right now, we remove the data at the beginning of list.
        """
        if self.size() > self.max_size:
            self._list = self._list[-self.max_size:]

    def size(self):
        return len(self._list)
    
    def sample_random_experiences(self, num_samples: int, seed: int = None):
        rng = np.random.default_rng(seed=seed)
        num_samples = min(num_samples, self.size()) #adjust num_samples to not exceed buffer size
        #choose the indices
        indices = rng.choice(range(self.size()), size=num_samples, replace=False)
        # use _get_components_by_index to get the components and return them
        obs, act, rew, don, nob = self._get_components_by_index(indices=indices.tolist())
        return (obs, act, rew, don, nob)
    
    def get_components(self):
        obs, act, rew, don, next_obs = self._get_components_by_index(indices=list(range(self.size())))
        return obs, act, rew, don, next_obs
    
    def _get_components_by_index(self, indices : List):
        """
        Returns the individual components of experiences chosen using 
        indices from self._list.
        :param List indices: The indices for experiences we pick out.
        """
        obs, act, rew, don, next_obs = (np.empty(shape=[len(indices)] + list(self.obs_shape), dtype=np.float32), 
                                        np.empty(shape=[len(indices)] + list(self.act_shape), dtype=np.float32),
                                        np.empty(shape=[len(indices)],                        dtype=np.float32),
                                        np.empty(shape=[len(indices)],                        dtype=np.bool8  ),
                                        np.empty(shape=[len(indices)] + list(self.obs_shape), dtype=np.float32))
        for i, idx in enumerate(indices):
            exp = self._list[idx]
            obs[i], act[i], rew[i], don[i], next_obs[i] = (exp.obs,
                                                           exp.action,
                                                           exp.reward,
                                                           exp.done,
                                                           exp.next_obs)
        return obs, act, rew, don, next_obs
    
    def shuffle(self):
        random.shuffle(self._list)

    def save(self, save_dir : str, file_name : str):
        raise Exception("ListBuffer will not support saving and loading simply because\
                        I am lazy to do so.")
    
    def load(self, path : str):
        raise Exception("ListBuffer will not support saving and loading simply because\
                        I am lazy to do so.")

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

    def __init__(self, max_size : int, obs_shape : Tuple[int], act_shape : Tuple[int]):
        """
        Initialize a numpy buffer with max_buffer_size capacity which we control
        based on 0-based indicing with the current and max sizes.
        :param int max_size: The maximum size of the buffer. We allocate this \
            size of memory when initializing this buffer.
        :param Tuple[int] obs_shape: The shape of the osbervation space.
        :param Tuple[int] act_shape: The shape of the action space.
        """
        self.obs      = np.empty(shape=[max_size] + list(obs_shape), dtype=np.float32)
        self.actions  = np.empty(shape=[max_size] + list(act_shape), dtype=np.float32)
        self.rewards  = np.empty(shape=[max_size],                   dtype=np.float32)
        self.dones    = np.empty(shape=[max_size],                   dtype=np.float32)
        self.next_obs = np.empty(shape=[max_size] + list(obs_shape), dtype=np.float32)
        
        self.obs_shape, self.act_shape = obs_shape, act_shape
        self._ptr, self._size, self.max_size = 0, 0, max_size
    
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
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = act
        self.rewards[self._ptr] = rew
        self.dones[self._ptr] = don
        self.next_obs[self._ptr] = next_obs
        # update size and ptr
        self._ptr = (self._ptr + 1) % self.max_size
        self._size = self._size + 1 if self._size < self.max_size else self._size
    
    def extend_buffer(self, buffer):
        """
        Extend the buffer using the content from another buffer.
        The result should be just like we appended to self all experiences 
        in buffer using append_experience (that is, we wrap around when we
        fill the entire buffer). 
        """
        """
        First, reach the state when the number of experiences we fill is smaller or equal to self.max_size
        To do so, we:
        0) check if buffer.size() + self.size() is greater than self.max_size - if so, wrap around occurs, and we go to 1)
           otherwise, wrap around does not occur, so we simply COPY_CONTENT
        1) calculate SKIPPING = buffer.size() - self.max_size which should be the number of addition 
           that won't matter since wrapping around will override them
        3) we determine the new pointer position after SKIPPING additions, which takes us to position
           (self.size() + SKIPPING) % self.max_size
        4) from this position, we add self.max_size as we wrap around, using COPY_CONTENT

        b1: 100 max, 30 filled
        b2: 400 max, 350 filled
        -> 1st, fill 70 using 350 -> 350 - 70 = 280 remaining
        -> then, fill 100 twice -> 280 - 200 = 80 remaining
        -> finally, fill by 80.
        This is really equivalent to:
        1) fill 250, but hypothetically -> new position will be (30 + 250) % 100 = 80
        2) then fill 20 with next 20, and finally wrap around and fill 80 for the last 80 
        """

        def _copy_to_buffer_from_i_to_j(i, j):
            self.obs[     self._ptr : self._ptr + j - i] = buffer.obs[i:j]
            self.actions[ self._ptr : self._ptr + j - i] = buffer.actions[i:j]
            self.rewards[ self._ptr : self._ptr + j - i] = buffer.rewards[i:j]
            self.dones[   self._ptr : self._ptr + j - i] = buffer.dones[i:j]
            self.next_obs[self._ptr : self._ptr + j - i] = buffer.next_obs[i:j]

        if (buffer.size() + self._ptr) <= self.max_size:
            _copy_to_buffer_from_i_to_j(0, buffer.size())
            self._ptr = (self._ptr + buffer.size()) % self.max_size
            self._size = min(self._size + buffer.size(), self.max_size)

        elif (buffer.size() + self._ptr) > self.max_size:
            if buffer.size() <= self.max_size: # if addign > self.buffer.max_size worth exps
                skipping = 0 # we skip no entry from buffer in that case
            else: #otherwise, we can skip everything that is not within the last self.max_size in buffer
                skipping = buffer.size() - self.max_size
            # update ptr at the point where ptr goes if we do all skipping additions
            self._ptr = (self._ptr + skipping) % self.max_size
            # replace the first half before wrapping around
            size_before_wrap = self.max_size - self._ptr
            _copy_to_buffer_from_i_to_j(skipping, skipping + size_before_wrap)
            self._ptr = 0
            _copy_to_buffer_from_i_to_j(skipping + size_before_wrap, buffer.size())
            self._ptr = buffer.size() - (skipping + size_before_wrap)
            self._size = self.max_size

    def size(self):
        """
        Returns the size of the buffer.
        """
        return self._size
    
    def get_components(self):
        """
        Returns the individual components of the buffer as numpy arrays in tuple:
        observation, action, reward, done, next observation.
        """
        return (self.obs[:self._size], self.actions[:self._size], self.rewards[:self._size], 
                self.dones[:self._size], self.next_obs[:self._size])
    
    def sample_random_experiences(self, num_samples : int, seed : int=None):
        """
        Returns num_samples random experiences taken from the buffer, returning
        numpy arrays for corresponding components.
        Takes in a seed for reproducibility.
        :param int num_samples: The number of samples we pick. The returned number of \
            samples is the minimum of this and the size of the buffer.
        :param int seed: The seed value useful for reproducible sampling. Not applied if None.
        :returns Tuple[np.ndarray]: Returns a tuple of numpy arrays corresponding to the transitions: \
            observations, actions, rewards, done flags and next observations.
        """
        rng = np.random.default_rng(seed=seed)
        num_samples = min(num_samples, self._size) #adjust num_samples to not exceed buffer size
        #choose the indices
        indices = rng.choice(range(self._size), size=num_samples, replace=False)
        # returns a copy as a result of fancy indexing as of right now
        return (self.obs[indices].copy(), self.actions[indices].copy(), self.rewards[indices].copy(), 
                self.dones[indices].copy(), self.next_obs[indices].copy())
    
    def shuffle(self):
        """
        Shuffles the buffer. As the only function that gets access based on indexing is
        get_components, this shall not be used as part of NdArrayBuffer. 
        """
        raise Exception("Shuffling of this buffer class is not supported; please use\
                        sample_random_experiences in order to get experiences from a shuffled buffer.")
    
    def save(self, save_dir : str, file_name : str):
        if not file_name.endswith(".npz"): file_name += ".npz"
        # create / overwrite the current numpy array object
        open(os.path.join(save_dir, file_name), 'w').close()

        np.savez_compressed(file=os.path.join(save_dir, file_name), 
                            obs=self.obs, actions=self.actions,
                            rewards=self.rewards, dones=self.dones, 
                            next_obs=self.next_obs,
                            obs_shape=np.array(self.obs_shape),
                            act_shape=np.array(self.act_shape),
                            _ptr=np.array(self._ptr),
                            _size=np.array(self._size),
                            max_size=np.array(self.max_size))

    def load(self, path : str):
        if not path.endswith(".npz"): path += ".npz"
        np_arrs = np.load(path)
        self.obs       = np_arrs["obs"]
        self.actions   = np_arrs["actions"]
        self.rewards   = np_arrs["rewards"]
        self.dones     = np_arrs["dones"]
        self.next_obs  = np_arrs["next_obs"]
        self.obs_shape = tuple(np_arrs["obs_shape"])
        self.act_shape = tuple(np_arrs["act_shape"])
        self._ptr      = int(np_arrs["_ptr"])
        self._size     = int(np_arrs["_size"])
        self.max_size  = int(np_arrs["max_size"])