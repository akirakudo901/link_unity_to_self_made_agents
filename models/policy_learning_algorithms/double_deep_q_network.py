"""
A DDQN algorithm to be used as learning algorithm.
"""

import os

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.trainers.utils.buffer import NdArrayBuffer
from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

class DoubleDeepQNetwork(PolicyLearningAlgorithm):
    ALGORITHM_NAME = "DDQN"

    class DNN(nn.Module):

        def __init__(self, 
                     input_size : int, 
                     output_size : int, 
                     dqn_layer_sizes : Tuple[int] = (8, 16)):
            # Initializes a new DNN.
            super(DoubleDeepQNetwork.DNN, self).__init__()

            self.fc_relu_stack = PolicyLearningAlgorithm.create_net(
                input_size=input_size,
                output_size=output_size,
                interim_layer_sizes=dqn_layer_sizes
            )
            
        def forward(self, x):
            x = self.fc_relu_stack(x)
            return x
    

    def __init__(self, 
                 obs_dim_size : int = None, act_num_discrete : int = None, 
                 l_r : float = 0.1, d_r : float = 0.95,
                 soft_update_coefficient : float = 0.005,
                 update_target_every_N_updates : int = 1,
                 dqn_layer_sizes : Tuple[int] = None,
                 env = None):
        """
        Initializes a DDQN algorithm. 
        *Either both of obs_dim_size and act_num_discrete, or env, should be given. If
        both are given, information automatically extracted from env is prioritized.
        If neither or only one of the first two are given, an error is raised.

        :param int obs_dim_size: The dimension size for observations.
        :param int act_num_discrete: The discrete numbers of actions (since this is Q-learning).
        :param float l_r: The learning rate of neural networks, defaults to 0.1
        :param float d_r: The decay rate in bootstrapping the Q-value, defaults to 0.95
        :param float soft_update_coefficient: Designates the proportions at which we 
        update the target network's weights utilizing that of the policy network. Defaults to 0.005.
        :param int update_target_every_N_updates: Designates once every how many steps the
        target network's weights are updated using that of the policy network. Defaults to 1.
        :param Tuple[int] dqn_layer_sizes: The number of neurons in each layer in the DQN modules.
        If not given, defaults to [8, 16, 16].
        :param env: The environment on which we will train this algorithm. Optionally passed, so that
        features about the environment are automatically extracted.
        * IF GIVEN, INPUTS SUCH AS obs_dim_size AND act_num_discrete ARE IGNORED!

        """
        if (obs_dim_size == None or act_num_discrete == None) and env == None:
            raise Exception("Either both of obs_dim_size and act_num_discrete, or env, " +
                             "should be given!")

        super().__init__(obs_dim_size=obs_dim_size, act_num_discrete=act_num_discrete, env=env)

        self.learning_rate = l_r
        self.discount = d_r
        self.soft_update_coefficient = soft_update_coefficient
        self.update_target_every_N_updates = update_target_every_N_updates
        
        self.dnn_policy = DoubleDeepQNetwork.DNN(
            input_size=self.obs_dim_size, 
            output_size=self.act_num_discrete,
            dqn_layer_sizes=dqn_layer_sizes
            ).to(self.device)
        self.dnn_target = DoubleDeepQNetwork.DNN(
            input_size=self.obs_dim_size, 
            output_size=self.act_num_discrete,
            dqn_layer_sizes=dqn_layer_sizes
            ).to(self.device)

        self.optim = optim.Adam(self.dnn_policy.parameters(), lr=self.learning_rate)

        self.dnn_update_counter = 0
        self.loss_history = [] # a List of floats
        
    def get_optimal_action(self, state):
        state_tensor = super().get_optimal_action(state)      
        state_tensor = state_tensor.to(self.device)
        prediction = self.dnn_policy(state_tensor).cpu()
        return torch.argmax(prediction).numpy()
    
    def save(self, task_name, save_dir=None):
        save_dir = PolicyLearningAlgorithm.get_saving_directory_name(
            task_name=task_name,
            algorithm_name=DoubleDeepQNetwork.ALGORITHM_NAME,
            save_dir=save_dir
            )
        
        save_dir += ".pth"
        torch.save(self.dnn_policy.state_dict(), save_dir)

    def _get_parameter_dict(self):
        """
        Returns a dictionary of the relevant parameters to be saved for 
        this algorithm, to track saving progress.

        :return Dict algorithm_param: The parameters of the algorithm that are saved.
        """
        algorithm_param = super()._get_parameter_dict()
        algorithm_param["learning_rate"] = self.learning_rate
        algorithm_param["discount"] = self.discount
        algorithm_param["soft_update_coefficient"] = self.soft_update_coefficient
        algorithm_param["update_target_every_N_updates"] = self.update_target_every_N_updates
        algorithm_param["dnn_update_counter"] = self.dnn_update_counter
        algorithm_param["loss_history"] = self.loss_history
        return algorithm_param

    def load(self, path):
        self.dnn_policy.load_state_dict(torch.load(path if path.endswith('.pth') else (path + '.pth')))
        # move to self.device
        self.dnn_policy.to(self.device)
    
    def _load_parameter_dict(self, dict: Dict):
        super()._load_parameter_dict(dict)
        self.__init__(obs_dim_size=self.obs_dim_size,
                      act_num_discrete=self.act_num_discrete,
                      l_r=dict["learning_rate"],
                      d_r=dict["discount"],
                      soft_update_coefficient=dict["soft_update_coefficient"],
                      update_target_every_N_updates=dict["update_target_every_N_updates"])
        self.dnn_update_counter = dict["dnn_update_counter"]
        self.loss_history       = dict["loss_history"]
        
    def update(self, buffer : NdArrayBuffer):
        """
        Updates the DDQN algorithms based on the given buffer.
        Returns the loss obtained in this update loop.

        :param NdArrayBuffer buffer: The buffer we use to update the algorithm.
        :return float loss: Returns the loss within this update.
        """
        BATCH_SIZE = 32

        if buffer.size() < BATCH_SIZE: return
        
        # sample a minibatch of transitions from the buffer
        np_obs, np_act, np_rew, np_don, np_next_obs = buffer.sample_random_experiences(num_samples=BATCH_SIZE) #TODO no seed, how to incorporate it?
        states      = torch.from_numpy(np_obs)
        actions     = torch.from_numpy(np_act)
        rewards     = torch.from_numpy(np_rew)
        dones       = torch.from_numpy(np_don)
        next_states = torch.from_numpy(np_next_obs)

        # calculate the actual Q values
        actual_q = (rewards +
                     (1 - dones) *
                     self.discount * 
                     torch.max(self.dnn_target(next_states).detach(), dim=1).values)
        
        # calculate the predicted Q values
        pred_q = self.dnn_policy(states)
        mask = torch.zeros(
            pred_q.shape
            ).scatter_(1, actions.unsqueeze(1).to(torch.int64), 1.)
        # apply mask to obtain the relevant predictions for current states
        compared_q = torch.sum(pred_q * mask, dim=1)
        
        # calculate loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(compared_q, actual_q)
        # propagate the result
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # update the target dnn appropriately after one update
        self._update_target()
        self.loss_history.append(loss.item())
        # log the result to wandb if we are using it
        if wandb.run is not None: wandb.log({"QNet Loss" : loss.item()})
            
    def _update_target(self):
        # do hard update
        if self.update_target_every_N_updates != None:
            self.dnn_update_counter += 1
            if self.dnn_update_counter % self.update_target_every_N_updates == 0:
                self.dnn_target.load_state_dict(self.dnn_policy.state_dict())
        # or soft update
        elif self.soft_update_coefficient != None:
            for param, target_param in zip(self.dnn_policy.parameters(), self.dnn_target.parameters()):
                target_param.data.copy_(self.soft_update_coefficient * 
                                        param.data + 
                                        (1 - self.soft_update_coefficient) * 
                                        target_param.data)
                
    def _delete_saved_algorithms(self, dir : str, task_name : str, training_id : int):
        save_dir = PolicyLearningAlgorithm.get_saving_directory_name(
            task_name=f"{task_name}_{training_id}",
            algorithm_name=DoubleDeepQNetwork.ALGORITHM_NAME,
            save_dir=f"{dir}/{task_name}_{training_id}"
            )
        save_dir += ".pth"
        os.remove(save_dir)
    
    def show_loss_history(self, task_name : str, save_figure : bool=True, save_dir : str=None):
        """
        Shows the loss history of the Q-network.

        :param str task_name: The name of the task, e.g. pendulum.
        :param bool save_figure: Whether to save the figure, defaults to True.
        :param str save_dir: The directory to which we save figures. Set to current
        directory if not given or None.
        """
        PolicyLearningAlgorithm.plot_history_and_save(history=self.loss_history, 
                                                         loss_source="Qnet", 
                                                         task_name=task_name, 
                                                         save_figure=save_figure, 
                                                         save_dir=save_dir)

# Below is code useful when discretizing continuous environments such that we can apply DDQN 
ACTION_DISCRETIZED_NUMBER = 100
MAX, MIN = 0.5, -0.5

@staticmethod
def convert_discrete_action_to_continuous(discrete_actions : torch.tensor) -> np.ndarray:
    """
    Input tensor is of shape (batch, 39) where each of the 39 values are an integer
    between 0 and (ACTION_DISCRETIZED_NUMBER - 1).
    We convert these back into floats of value between MAX and MIN. 
    """
    continuous_actions = discrete_actions * ((MAX - MIN) / ACTION_DISCRETIZED_NUMBER) + MIN
    return continuous_actions

@staticmethod
def convert_continuous_action_to_discrete(continuous_actions : torch.tensor) -> np.ndarray:
    """
    Input tensor is of shape (batch, 39) where each of the 39 values are floats between MIN 
    and MAX. We convert these into value between 0 and (ACTION_DISCRETIZED_NUMBER = 1).
    """
    discrete_actions = (continuous_actions - MIN) // ((MAX - MIN) / ACTION_DISCRETIZED_NUMBER)
    return discrete_actions