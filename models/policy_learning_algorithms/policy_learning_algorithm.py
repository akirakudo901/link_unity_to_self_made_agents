"""
Base abstract class for whatever policy learning algorithm we use.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
import logging
from math import log2
import os
import traceback
from typing import Dict, List, Tuple, Union

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import torch.nn as nn

from models.trainers.utils.buffer import Buffer

class PolicyLearningAlgorithm(ABC):

    ALGORITHM_SAVE_DIR = "trained_algorithms"

    @staticmethod
    def create_net(input_size : int, 
                   output_size : int, 
                   interim_layer_sizes : Tuple[int]):
        """
        Creates a network model which input and output sizes as well as
        interim layer sizes are determined. Each layer will be linear
        followed by ReLU except the very last layer which lacks activation.
        Returns the generated network as instance of nn.Sequential.

        :param int input_size: The number of dimensions for the input.
        :param int output_size: The number of dimensions for the output.
        :param Tuple[int] interim_layer_sizes: The number of neurons for each \
        interim layer.
        :return nn.Sequential: The generated model as instance of nn.Sequential.
        """
        layer_sizes = [input_size] + list(interim_layer_sizes) + [output_size]
        layers = []
        for i, sz in enumerate(layer_sizes[:-1]):
            layers.append(nn.Linear(sz, layer_sizes[i+1]))
            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

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
    
    @abstractmethod
    def _get_parameter_dict(self):
        """
        Returns a dictionary of the relevant parameters to be saved for 
        this algorithm, to track saving progress.

        :return Dict algorithm_param: The parameters of the algorithm that are saved.
        """
        algorithm_param = {
            "obs_dim_size" : self.obs_dim_size,
            "act_dim_size" : self.act_dim_size,
            "obs_num_discrete" : self.obs_num_discrete,
            "act_num_discrete" : self.act_num_discrete,
            "obs_ranges" : self.obs_ranges,
            "act_ranges" : self.act_ranges,
        }
        return algorithm_param

    @abstractmethod
    def load(self, path : str):
        """
        Loads the current policy.

        :param str path: Path from which we load the algorithm.
        """
        pass
    
    @abstractmethod
    def _load_parameter_dict(self, dict : Dict):
        """
        *CALLING THIS FUNCTION WILL REINITIALIZE SELF!!
        Loads the dictionary containing relevant parameters for 
        this algorithm while loading previous progress.

        :param Dict dict: Dictionary of parameters for the algorithm getting loaded.
        """
        self.obs_dim_size = dict["obs_dim_size"]
        self.act_dim_size = dict["act_dim_size"]
        self.obs_num_discrete = dict["obs_num_discrete"]
        self.act_num_discrete = dict["act_num_discrete"]
        self.obs_ranges = dict["obs_ranges"]
        self.act_ranges = dict["act_ranges"]

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

    @abstractmethod
    def _delete_saved_algorithms(self, dir : str, task_name : str, training_id : int):
        """
        Deletes the saved algorithm files.
        
        :param str dir: String specifying the path in which the deleted data is held.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        pass

    ########################
    # Non-abstract methods

    def __call__(self, state : Union[torch.tensor, np.ndarray]):
        """
        A raw call to the function which then calls get_optimal_action.

        :param torch.tensor or np.ndarray state: The state of observation in question 
        to which this algorithm was applied.
        """
        return self.get_optimal_action(state)
    
    def save_training_progress(self, dir : str, task_name : str, training_id : int):
        """
        Saves training progress info specific to the algorithm in the given directory.

        :param str dir: String specifying the path to which we save.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        def try_saving_except(call, saving_the___successfully : str, *args, **kwargs):
            try:
                # print(f"Saving the {saving_the___successfully}...", end="")
                call(*args, **kwargs)
                # print(f"successful.")
            except Exception:
                print(f"\nSome exception occurred while saving the {saving_the___successfully}...")
                logging.error(traceback.format_exc())
        
        # save the algorithm in the below directory for tracking progress
        try_saving_except(self.save, saving_the___successfully="algorithm parameters", 
                          task_name=f"{task_name}_{training_id}", 
                          save_dir=f"{dir}/{task_name}_{training_id}")

        # save the yamlirized features in this algorithm
        def save_yaml():
            param_dict = self._get_parameter_dict()
            with open(f"{dir}/{task_name}_{training_id}_Algorithm_Param.yaml",
                    'w') as yaml_file:
                yaml.dump(param_dict, yaml_file)
        
        try_saving_except(save_yaml, saving_the___successfully="algorithm fields")
    
    def load_training_progress(self, dir : str, task_name : str, training_id : int):
        """
        Loads training progress info specific to the algorithm from the given directory.
        *This function uses load() instead of safe_load() from PyYaml.
        This should be safe so far as we only load files created by this code;
        if you do import codes from the outside, beware of YAML's building 
        functionality, which builds classes others have defined that might be harmful.
        
        :param str dir: String specifying the path from which we load.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        def try_loading_except(call, loading_the___successfully : str, *args, **kwargs):
            try:
                # print(f"Loading the {loading_the___successfully}...", end="")
                call(*args, **kwargs)
                # print(f"successful.")
            except Exception:
                print(f"\nSome exception occurred while loading the {loading_the___successfully}...")
                logging.error(traceback.format_exc())
        # !! BELOW, ORDER MATTERS AS CALLING _LOAD_PARAMETER_DICT REINITIALIZES THE ALGORITHM!
        # load the yamlirized features in this algorithm
        def load_yaml():
            with open(f"{dir}/{task_name}_{training_id}_Algorithm_Param.yaml",
                    'r') as yaml_file:
                #should be safe so far as we only load files created by this code;
                # if you do import codes from the outside, beware of YAML's 
                # building functionality that might be harmful.
                yaml_dict = yaml.load(yaml_file, Loader=yaml.Loader) 
                self._load_parameter_dict(dict=yaml_dict)
        
        try_loading_except(load_yaml, loading_the___successfully="algorithm fields")
        
        # load the algorithm to track progress
        try_loading_except(self.load, loading_the___successfully="algorithm parameters", 
                          path=f"{dir}/{task_name}_{training_id}")
    
    def delete_training_progress(self, dir : str, task_name : str, training_id : int):
        """
        Deletes training progress info specific to the algorithm.
        
        :param str dir: String specifying the path in which the deleted info is stored.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        self._delete_saved_algorithms(task_name=task_name, training_id=training_id)
        os.remove(f"{dir}/{task_name}_{training_id}_Algorithm_Param.yaml")

    ################
    # Static methods

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

        def plot_figure(h, fig_name, loss_type, y_range=None):
            plt.clf()
            plt.title(f"{task_name} {loss_source} {loss_type}Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            if y_range != None: plt.ylim(y_range)
            plt.plot(range(0, len(h)), h)
            if save_figure: plt.savefig(f"{save_dir}/{fig_name}")
            plt.show()

        if save_dir == None: save_dir = "."
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        # if difference is too big, create log, twice_std and
        minimum = min(history)
        if (max(history) - minimum) >= 200:

            # 1 - Log loss
            # ensure that loss is greater than 0
            history = [log2(history[i]) if minimum > 0
                       else log2(history[i] - minimum + 1e-6) - log2(abs(minimum))
                       for i in range(len(history))]
            figure_name = f"{task_name}_{loss_source}_logLoss_history_fig.png"
            plot_figure(history, figure_name, "Log")

            # 2 - STD loss
            # show mean +- std*2
            mean = sum(history) / len(history)
            std = np.std(history)
            interval = std * 4
            figure_name = f"{task_name}_{loss_source}_stdLoss_history_fig.png"
            plot_figure(history, figure_name, "Std", y_range=[mean - interval, mean + interval])

            # 3 - Set interval loss
            # show minimum + 10
            figure_name = f"{task_name}_{loss_source}_setIntervalLoss_history_fig.png"
            plot_figure(history, figure_name, "Set Interval", y_range=[minimum, minimum + 10])

        # otherwise simply plot the result
        else:
            figure_name = f"{task_name}_{loss_source}_loss_history_fig.png"
            plot_figure(history, figure_name, "")

    @staticmethod
    def set_device():
        """
        Returns either a cpu or cuda device depending on availability.

        :return torch.device device: The device that was found available.        
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        return device
    
# OTHER USEFUL FUNCTIONS

# EXPLORATION FUNCTIONS
def no_exploration(actions : Union[np.ndarray, torch.tensor], env):
    """
    An exploration function that takes an action and return it while
    attempting to convert it into a np.ndarray.
    To be passed to trainers as arguments.

    :param np.ndarray or torch.tensor actions: The action to be passed.
    :param env: An environment to be passed - is a placeholder.
    :raises Exception: Raises an exception if actions is of neither type specified above.
    :return np.ndarray actions: Returns an np.ndarray version of the action passed.
    """
    # since exploration is inherent in SAC, we don't need epsilon to do anything
    if type(actions) == type(torch.tensor([0])):
        return actions.detach().numpy()
    elif type(actions) == type(np.array([0])):
        return actions
    else:
        raise Exception("Value passed as action to no_exploration was of type ", type(actions), 
                        "but should be either a torch.tensor or np.ndarray to successfully work.") 

# USEFUL FOR GENERATING SETS OF HYPERPARAMETERS
def generate_parameters(default_parameters : Dict, default_name : str, **kwargs):
    """
    Generate new parameters which are combinations of the
    given keyword arguments. The keyword arguments can 
    either be:
    - a single value to be put in all parameters
    - a list of all parameters you want to try

    e.g. kwargs = { "learning_rate" : 1e-2, "discount_rate" : [0.99, 0.95, 0.90] }

    :param Dict default_parameters: The dictionary holding all \
    default parameters for the parameter sets.
    :param Dict default_name: The name characterizing the default dictionary.
    :return Dict returned: A dictionary holding all generated parameter dicts \
    paired with auto-generated names attributed to them.
    """

    def new_dict_from_old(old_name, old_dict, key, val):
        if old_name == default_name:
            new_name = f"{key}_{str(val)}"
        else:
            new_name = name + f"_{key}_{str(val)}"
        old_dict[key] = val
        return new_name, old_dict

    returned = {default_name : default_parameters}
    for key, values in kwargs.items():
        if type(values) != type([]):
            for d in returned.values(): d[key] = values
        elif type(values) == type([]) and len(values) == 0:
            pass
        elif type(values) == type([]):
            new_dicts = {}
            for name, d in returned.items():
                new_name, new_dict = new_dict_from_old(old_name=name, 
                                                       old_dict=d, 
                                                       key=key, 
                                                       val=values[0])
                new_dicts[new_name] = new_dict 
                
                for v in values[1:]:
                    new_d = deepcopy(d)
                    new_name, new_dict = new_dict_from_old(old_name=name,
                                                           old_dict=new_d,
                                                           key=key,
                                                           val=v)
                    new_dicts[new_name] = new_dict
            returned = new_dicts
    return returned

def generate_name_from_parameter_dict(parameter_dict : Dict):
    """
    Generates a name characterizing a given parameter dict.
    The order of terms in the dictionary is the order in which 
    parameters are listed.

    :param Dict parameter_dict: The parameter dict for which we generate the name.
    """
    acc = str(list(parameter_dict.keys())[0]) + "_" + str(list(parameter_dict.values())[0])
    [acc := acc + "_" + str(key) + "_" + str(val) for key, val in list(parameter_dict.items())[1:]]
    return acc
        