
import unittest

from torch import nn

from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

class TestPolicyLearningAlgorithm(unittest.TestCase):

    ##################
    # NORMAL METHODS #
    ##################  
    
    # TODO
    def test_save_training_progress(self):
        """
        Saves training progress info specific to the algorithm in the given directory.

        :param str dir: String specifying the path to which we save.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        # def try_saving_except(call, saving_the___successfully : str, *args, **kwargs):
        #     try:
        #         # print(f"Saving the {saving_the___successfully}...", end="")
        #         call(*args, **kwargs)
        #         # print(f"successful.")
        #     except Exception:
        #         print(f"\nSome exception occurred while saving the {saving_the___successfully}...")
        #         logging.error(traceback.format_exc())
        
        # # save the algorithm in the below directory for tracking progress
        # try_saving_except(self.save, saving_the___successfully="algorithm parameters", 
        #                   task_name=f"{task_name}_{training_id}", 
        #                   save_dir=f"{dir}/{task_name}_{training_id}")

        # # save the yamlirized features in this algorithm
        # def save_yaml():
        #     param_dict = self._get_parameter_dict()
        #     with open(f"{dir}/{task_name}_{training_id}_Algorithm_Param.yaml",
        #             'w') as yaml_file:
        #         yaml.dump(param_dict, yaml_file)
        
        # try_saving_except(save_yaml, saving_the___successfully="algorithm fields")
    
    # TODO
    def test_load_training_progress(self):
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
        # def try_loading_except(call, loading_the___successfully : str, *args, **kwargs):
        #     try:
        #         # print(f"Loading the {loading_the___successfully}...", end="")
        #         call(*args, **kwargs)
        #         # print(f"successful.")
        #     except Exception:
        #         print(f"\nSome exception occurred while loading the {loading_the___successfully}...")
        #         logging.error(traceback.format_exc())
        # # !! BELOW, ORDER MATTERS AS CALLING _LOAD_PARAMETER_DICT REINITIALIZES THE ALGORITHM!
        # # load the yamlirized features in this algorithm
        # def load_yaml():
        #     with open(f"{dir}/{task_name}_{training_id}_Algorithm_Param.yaml",
        #             'r') as yaml_file:
        #         #should be safe so far as we only load files created by this code;
        #         # if you do import codes from the outside, beware of YAML's 
        #         # building functionality that might be harmful.
        #         yaml_dict = yaml.load(yaml_file, Loader=yaml.Loader) 
        #         self._load_parameter_dict(dict=yaml_dict)
        
        # try_loading_except(load_yaml, loading_the___successfully="algorithm fields")
        
        # # load the algorithm to track progress
        # try_loading_except(self.load, loading_the___successfully="algorithm parameters", 
        #                   path=f"{dir}/{task_name}_{training_id}")
    
    # TODO
    def test_generate_parameters(self):
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

        # def new_dict_from_old(old_name, old_dict, key, val):
        #     if old_name == default_name:
        #         new_name = f"{key}_{str(val)}"
        #     else:
        #         new_name = name + f"_{key}_{str(val)}"
        #     old_dict[key] = val
        #     return new_name, old_dict

        # returned = {default_name : default_parameters}
        # for key, values in kwargs.items():
        #     if type(values) != type([]):
        #         for d in returned.values(): d[key] = values
        #     elif type(values) == type([]) and len(values) == 0:
        #         pass
        #     elif type(values) == type([]):
        #         new_dicts = {}
        #         for name, d in returned.items():
        #             new_name, new_dict = new_dict_from_old(old_name=name, 
        #                                                 old_dict=d, 
        #                                                 key=key, 
        #                                                 val=values[0])
        #             new_dicts[new_name] = new_dict 
                    
        #             for v in values[1:]:
        #                 new_d = deepcopy(d)
        #                 new_name, new_dict = new_dict_from_old(old_name=name,
        #                                                     old_dict=new_d,
        #                                                     key=key,
        #                                                     val=v)
        #                 new_dicts[new_name] = new_dict
        #         returned = new_dicts
        # return returned

    # TODO
    def test_generate_name_from_parameter_dict(self):
        """
        Generates a name characterizing a given parameter dict.
        The order of terms in the dictionary is the order in which 
        parameters are listed.

        :param Dict parameter_dict: The parameter dict for which we generate the name.
        """
        # acc = str(list(parameter_dict.keys())[0]) + "_" + str(list(parameter_dict.values())[0])
        # [acc := acc + "_" + str(key) + "_" + str(val) for key, val in list(parameter_dict.items())[1:]]
        # return acc
            
    ##################
    # STATIC METHODS #
    ##################  
    
    def test_create_net(self):
        def verify_correct_layer_size(layer_sizes, layers):
            for i, l in enumerate(layers):
                layersz_idx = i // 2
                if i % 2 == 0:
                    self.assertEqual(l.in_features, layer_sizes[layersz_idx])
                    self.assertEqual(l.out_features, layer_sizes[layersz_idx + 1])
                else:
                    self.assertTrue(isinstance(l, nn.ReLU))
                    
        net1 = PolicyLearningAlgorithm.create_net(input_size=3, output_size=5, interim_layer_sizes=(8, 32))
        layer_sizes = (3, 8, 32, 5)
        verify_correct_layer_size(layer_sizes, net1)

        net2 = PolicyLearningAlgorithm.create_net(input_size=45, output_size=2, interim_layer_sizes=(10, 20, 30))
        layer_sizes2 = (45, 10, 20, 30, 2)
        verify_correct_layer_size(layer_sizes2, net2)

        net3 = PolicyLearningAlgorithm.create_net(input_size=4, output_size=1, interim_layer_sizes=(64, 64))
        layer_sizes3 = (4, 64, 64, 1)
        verify_correct_layer_size(layer_sizes3, net3)

    # TODO
    def test_get_gym_environment_specs(self):
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
        # def get_specs_of_space(space):
        #     if isinstance(space, gymnasium.spaces.Text):
        #         raise Exception("Behavior against gym Text spaces is not well-defined.")
        #     elif isinstance(space, gymnasium.spaces.Box): #env should be flattened outside
        #         dim_size = gymnasium.spaces.utils.flatdim(space)
        #         num_discrete = None
        #         ranges = tuple([(space.low[i], space.high[i]) 
        #                             for i in range(dim_size)])
        #     elif isinstance(space, gymnasium.spaces.MultiBinary):
        #         dim_size = gymnasium.spaces.utils.flatdim(space)
        #         num_discrete, ranges = None, None
        #     elif isinstance(space, gymnasium.spaces.Discrete):
        #         dim_size = 1 #assuming discrete states are input as distinct integers to nn
        #         num_discrete = space.n
        #         ranges = None
        #     elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        #         dim_size = len(space.nvec)
        #         num_discrete = sum(space.nvec)
        #         ranges = None
        #     return dim_size, num_discrete, ranges

        # # dimension size of observation - needed to initialize policy input size
        # obs_dim_size, obs_num_discrete, obs_ranges = get_specs_of_space(env.observation_space)
        
        # # dimension size of action - needed to initialize policy output size
        # # also returns the number of discrete actions for discrete environments,
        # # or None for continuous.
        # act_dim_size, act_num_discrete, act_ranges = get_specs_of_space(env.action_space)
        
        # return { 
        #     "obs_dim_size" : obs_dim_size, "act_dim_size" : act_dim_size, 
        #     "obs_num_discrete" : obs_num_discrete, "act_num_discrete" : act_num_discrete,
        #     "obs_ranges" : obs_ranges, "act_ranges" : act_ranges
        #     }
    
    # TODO
    def test_plot_history_and_save(self):
        """
        Plots and saves a given loss history. If the max & min difference of loss is
        greater than 200, we apply log to make it more visible.

        :param List[float] history: A loss history expressed as list of floats.
        :param str loss_source: Where the loss comes from, e.g. qnet1.
        :param str task_name: The general task name, e.g. pendulum.
        :param bool save_figure: Whether to save the figures or not.
        :param str save_dir: The directory to which we save figures.
        """

        # def plot_figure(h, fig_name, loss_type, y_range=None):
        #     plt.clf()
        #     plt.title(f"{task_name} {loss_source} {loss_type}Loss")
        #     plt.xlabel("Epochs")
        #     plt.ylabel("Loss")
        #     if y_range != None: plt.ylim(y_range)
        #     plt.plot(range(0, len(h)), h)
        #     if save_figure: plt.savefig(f"{save_dir}/{fig_name}")
        #     plt.show()

        # if save_dir == None: save_dir = "."
        # if not os.path.exists(save_dir): os.mkdir(save_dir)

        # # if difference is too big, create log, twice_std and
        # minimum = min(history)
        # if (max(history) - minimum) >= 200:

        #     # 1 - Log loss
        #     # ensure that loss is greater than 0
        #     history = [log2(history[i]) if minimum > 0
        #                else log2(history[i] - minimum + 1e-6) - log2(abs(minimum))
        #                for i in range(len(history))]
        #     figure_name = f"{task_name}_{loss_source}_logLoss_history_fig.png"
        #     plot_figure(history, figure_name, "Log")

        #     # 2 - STD loss
        #     # show mean +- std*2
        #     mean = sum(history) / len(history)
        #     std = np.std(history)
        #     interval = std * 4
        #     figure_name = f"{task_name}_{loss_source}_stdLoss_history_fig.png"
        #     plot_figure(history, figure_name, "Std", y_range=[mean - interval, mean + interval])

        #     # 3 - Set interval loss
        #     # show minimum + 10
        #     figure_name = f"{task_name}_{loss_source}_setIntervalLoss_history_fig.png"
        #     plot_figure(history, figure_name, "Set Interval", y_range=[minimum, minimum + 10])

        # # otherwise simply plot the result
        # else:
        #     figure_name = f"{task_name}_{loss_source}_loss_history_fig.png"
        #     plot_figure(history, figure_name, "")