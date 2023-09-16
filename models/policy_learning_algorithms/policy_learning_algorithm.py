"""
Base abstract class for whatever policy learning algorithm we use.
"""

from abc import ABC, abstractmethod

class OffPolicyLearningAlgorithm(ABC):

    @abstractmethod
    def __init__(self, observation_dim_size, action_dim_size):
        """
        A generic learning algorithm initializer.

        :param int observation_dim_size: The dimension size of the observation space. 
        :param int action_dim_size: The dimension size of the action space.
        """
        self.obs_dim_size = observation_dim_size
        self.act_dim_size = action_dim_size
    
    @abstractmethod
    def update(self, experiences):
        """
        Updates the algorithm according to a buffer of experience.
        """
        pass
    
    @abstractmethod
    def get_optimal_action(self, state):
        """
        Given the state, returns the corresponding optimal action under current knowledge.

        :param state: The state for which we return the optimal action.
        """
        pass

    @abstractmethod
    def save(self, task_name : str):
        """
        Saves the current policy.

        :param str task_name: The name of the task according to which we save the algorithm.
        """
        pass
    
    @abstractmethod
    def load(self, path : str):
        """
        Loads the current policy.

        :param str path: Path from which we load the algorithm.
        """
        pass