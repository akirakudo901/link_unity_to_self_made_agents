"""
Base abstract class for whatever policy learning algorithm we use.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from math import log2
from typing import List, Tuple, Union

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.trainers.utils.buffer import Buffer

class PolicyLearningAlgorithm(ABC):

    ALGORITHM_SAVE_DIR = "trained_algorithms"

    PROGRESS_SAVING_DIR = "trained_algorithms/_in_progress"

    @abstractmethod
    def __init__(self, 
                 obs_dim_size : int=None, act_dim_size : int=None, 
                 obs_num_discrete : int=None, act_num_discrete : int=None, 
                 obs_ranges : Tuple[Tuple[float]]=None, 
                 act_ranges : Tuple[Tuple[float]]=None,
                 env=None
                 ):
        """
        A default algorithm initializer & setter.
        If an env of compatible type is passed, the arguments are obtained 
        and set from the environment, and all other arguments passed are ignored.
        Compatible types:
        - gymnasium environment

        :param int obs_dim_size: The observation dimension size, defaults to None
        :param int act_dim_size: The action dimension size, defaults to None
        :param int obs_num_discrete: The number of discrete observation for discrete
        type environments, defaults to None
        :param int act_num_discrete: The above for actions, defaults to None
        :param Tuple[Tuple[float]] obs_ranges: The ranges [low, high] of each observation
        dimension for a continuous environment, defaults to None
        :param Tuple[Tuple[float]] act_ranges: The above for actions, defaults to None
        :param env: The environment from which we extract info, defaults to None
        """
        self.device = self.set_device()

        if env == None: pass
        elif isinstance(env, gymnasium.Env):
            spec_dict = PolicyLearningAlgorithm.get_gym_environment_specs(env)
            obs_dim_size = spec_dict["obs_dim_size"]
            act_dim_size = spec_dict["act_dim_size"]
            obs_num_discrete = spec_dict["obs_num_discrete"]
            act_num_discrete = spec_dict["act_num_discrete"]
            obs_ranges = spec_dict["obs_ranges"]
            act_ranges = spec_dict["act_ranges"]
        
        self.obs_dim_size = obs_dim_size
        self.act_dim_size = act_dim_size
        self.obs_num_discrete = obs_num_discrete
        self.act_num_discrete = act_num_discrete
        self.obs_ranges = obs_ranges
        self.act_ranges = act_ranges 
    
    @abstractmethod
    def update(self, experiences : Buffer):
        """
        Updates the algorithm according to a buffer of experience.
        :param Buffer experiences: The buffer from which we sample experiences.
        """
        pass
    
    @abstractmethod
    def get_optimal_action(self, state : Union[torch.tensor, np.ndarray]):
        """
        *This function itself can take an input and check if it is torch.tensor or
        np.ndarray. If one of the two, returns a torch.tensor to be passed to the policy.
        If not, it raises an error.

        *Function implemented in subclasses:
        Given the state, returns the corresponding optimal action under current knowledge.

        :param torch.tensor or np.ndarray state: The state for which we return the optimal action.
        """
        if type(state) == type(np.array([0])):
            state = torch.from_numpy(state)
        elif type(state) != type(torch.tensor([0])):
            raise Exception("State passed to get_optimal_action should be a np.array or torch.tensor.")
        return state

    @abstractmethod
    def save(self, task_name : str, save_dir : str):
        """
        Saves the current policy.

        :param str task_name: The name of the task according to which we save the algorithm.
        :param str save_dir: The directory to which this policy is saved.
        """
        pass

    @staticmethod
    def get_saving_directory_name(task_name : str, algorithm_name : str, save_dir : str):
        """
        Returns either save_dir if it is not None, or a custom name created from task_name.

        :param str task_name: The name of the task, e.g. pendulum.
        :param str algorithm_name: The name of the algorithm, e.g. SAC.
        :param str save_dir: The directory we save algorithms to using save().
        """
        creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if save_dir is None:
            save_dir = (f"{PolicyLearningAlgorithm.ALGORITHM_SAVE_DIR}/" + 
                        f"{algorithm_name}/{task_name}_{creation_time}")
        return save_dir
    
    # @abstractmethod
    def save_training_progress(task_name : str, training_id : int):
        """
        Abstract function which saves training progress info 
        specific to the algorithm.

        :param str task_name: name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        pass


    @abstractmethod
    def load(self, path : str):
        """
        Loads the current policy.

        :param str path: Path from which we load the algorithm.
        """
        pass

    @abstractmethod
    def show_loss_history(self, task_name : str, save_figure : bool=True, save_dir : str=None):
        """
        Plots figure indicating the appropriate loss to a network. Saves the resulting
        figures under save_dir, if save_figure is True. 

        :param str task_name: The name of the task we are working with.
        :param bool save_figure: Whether to save the figure plots as pngs
        in the current directory. Defaults to True.
        :param str save_dir: Directory to which we save the figures. If not given,
        we save the figures to the current directory.
        """
        pass


    def __call__(self, state : Union[torch.tensor, np.ndarray]):
        """
        A raw call to the function which then calls get_optimal_action.

        :param torch.tensor or np.ndarray state: The state of observation in question 
        to which this algorithm was applied.
        """
        return self.get_optimal_action(state)

    @staticmethod
    def plot_history_and_save(history : List[float], loss_source : str, 
                              task_name : str, save_figure : bool, save_dir : str):
        """
        Plots and saves a given loss history. If the max & min difference of loss is
        greater than 200, we apply log to make it more visible.

        :param List[float] history: A loss history expressed as list of floats.
        :param str loss_source: Where the loss comes from, e.g. qnet1.
        :param str task_name: The general task name, e.g. pendulum.
        :param bool save_figure: Whether to save the figures or not.
        :param str save_dir: The directory to which we save figures.
        """
        if save_dir == None: save_dir = "."
        figure_name = f"{task_name}_{loss_source}_loss_history_fig.png"

        # if difference is too big, apply log
        minimum = min(history)
        if (max(history) - minimum) >= 200:
            # ensure that loss is greater than 0
            history = [log2(history[i]) if minimum > 0 
                       else log2(history[i] - minimum + 1e-6) - log2(abs(minimum)) 
                       for i in range(len(history))]
            figure_name = f"{task_name}_{loss_source}_logLoss_history_fig.png"

        plt.clf()
        plt.plot(range(0, len(history)), history)
        if save_figure: plt.savefig(f"{save_dir}/{figure_name}")
        plt.show()

    @staticmethod
    def set_device():
        """
        Returns either a cpu or cuda device depending on availability.

        :return torch.device device: The device that was found available.        
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        return device
    
    @staticmethod
    def get_gym_environment_specs(env : gymnasium.Env):
        """
        Returns specs about an environment useful to initialize
        learning algorithms. Raises error on Text spaces.

        :param gymnasium.Env env: The environment we train the algorithm on.
        :returns dict: Returns a dictionary holding info about the environment:
        - obs_dim_size, the dimension size of observation space
        - act_dim_size, the dimension size of action space
        - obs_num_discrete, the number of discrete observations with Discrete & 
          MultiDiscrete spaces or None for Box & MultiBinary
        - act_num_discrete, the above for actions
        - obs_ranges, the range of observations with Box observations or None for
          MultiBinary & Discrete & MultiDiscrete
        - act_ranges, the above for actions
        """
        def get_specs_of_space(space):
            if isinstance(space, gymnasium.spaces.Text):
                raise Exception("Behavior against gym Text spaces is not well-defined.")
            elif isinstance(space, gymnasium.spaces.Box): #env should be flattened outside
                dim_size = gymnasium.spaces.utils.flatdim(space)
                num_discrete = None
                ranges = tuple([(space.low[i], space.high[i]) 
                                    for i in range(dim_size)])
            elif isinstance(space, gymnasium.spaces.MultiBinary):
                dim_size = gymnasium.spaces.utils.flatdim(space)
                num_discrete, ranges = None, None
            elif isinstance(space, gymnasium.spaces.Discrete):
                dim_size = 1 #assuming discrete states are input as distinct integers to nn
                num_discrete = space.n
                ranges = None
            elif isinstance(space, gymnasium.spaces.MultiDiscrete):
                dim_size = len(space.nvec)
                num_discrete = sum(space.nvec)
                ranges = None
            return dim_size, num_discrete, ranges

        # dimension size of observation - needed to initialize policy input size
        obs_dim_size, obs_num_discrete, obs_ranges = get_specs_of_space(env.observation_space)
        
        # dimension size of action - needed to initialize policy output size
        # also returns the number of discrete actions for discrete environments,
        # or None for continuous.
        act_dim_size, act_num_discrete, act_ranges = get_specs_of_space(env.action_space)
        
        return { 
            "obs_dim_size" : obs_dim_size, "act_dim_size" : act_dim_size, 
            "obs_num_discrete" : obs_num_discrete, "act_num_discrete" : act_num_discrete,
            "obs_ranges" : obs_ranges, "act_ranges" : act_ranges
            }