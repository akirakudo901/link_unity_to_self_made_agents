"""
Base abstract class for whatever agent we use which combines environment with learning algorithm.
"""

from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def reset(self):
        """
        # Resets the environment.
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Takes a step in the environment with the given action, and returns:
         - the resulting next state
         - the reward as a result of the action
         - if the episode was corretly terminated (boolean; terminated)
         - if the episode was incorrectly terminated (boolean; truncated)
         - additional info
    
        :param action: An object expressing the action chosen.
        """
        # return info
        pass
    
    @abstractmethod
    def update(self):
        """
        Updates the algorithm accordingly.
        """
        pass
    
    @abstractmethod
    def get_optimal_action(self, state):
        """
        Chooses the optimal action given the state object.

        :param state: The state for which we return the optimal action.
        """
        pass

    @abstractmethod 
    def get_random_action(self, state):
        """
        Gets a random action in the action space; has to be re-defined for each action.

        :param state: The state for which we return a random action.
        """
        pass