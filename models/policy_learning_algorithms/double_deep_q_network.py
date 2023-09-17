"""
A DDQN algorithm to be used as learning algorithm.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.trainers.gym_base_trainer import NdArrayBuffer

DNN_SAVE_FOLDER = "trained_algorithms/DDQN"

class DoubleDeepQNetwork:

    class DNN(nn.Module):

        def __init__(self, input_size, output_size):
            # Initializes a new DNN.
            super(DoubleDeepQNetwork.DNN, self).__init__()
            
            self.fc_relu_stack = nn.Sequential(
                nn.Linear(input_size, 8),
                nn.Sigmoid(),
                nn.Linear(8, 16),
                nn.Sigmoid(),
                nn.Linear(16, output_size),
            )

        def forward(self, x):
            x = self.fc_relu_stack(x)
            return x
    

    def __init__(self, 
                 observation_dim_size : int, discrete_action_size : int, 
                 l_r : float = 0.1, d_r : float = 0.95,
                 hard_update_every_N_updates : int = None,
                 soft_update_coefficient : float = None):
        """
        Initializes a DDQN algorithm.
        One can choose between hard and soft target update by giving one argument to 
        either hard_update_every_N_updates or soft_update_coefficient; if 
        both or neither arguments are given, this results in an error.

        :param int observation_dim_size: The dimension size for observations.
        :param int discrete_action_size: The dimension size for actions.
        :param float l_r: The learning rate of neural networks, defaults to 0.1
        :param float d_r: The decay rate in bootstrapping the Q-value, defaults to 0.95
        :param int hard_update_every_N_updates: Designates once every how many steps the
        target network's weights are updated to match that of the policy network, defaults to None.
        :param float soft_update_coefficient: Designates the proportions at which we at every step 
        update the target network's weights utilizing that of the policy network, defaults to None.
        """

        if hard_update_every_N_updates == None and soft_update_coefficient == None:
            raise Exception("Neither hard_update_every_N_updates nor soft_update_coefficient " +
                            "were specified - please specify exactly one of the two.")
        elif hard_update_every_N_updates != None and soft_update_coefficient != None:
            raise Exception("Both hard_update_every_N_updates and soft_update_coefficient " +
                            "were specified - please specify exactly one of the two.")
        else:
            if hard_update_every_N_updates != None:
                try: self.hard_update_every_N_updates = int(hard_update_every_N_updates)
                except: raise Exception("hard_update_every_N_updates must be an integer!")
                self.soft_update_coefficient = soft_update_coefficient
            elif soft_update_coefficient != None:
                self.hard_update_every_N_updates = hard_update_every_N_updates
                try: self.soft_update_coefficient = float(soft_update_coefficient)
                except: raise Exception("soft_update_coefficient must be a float!")
                
            
        self.discrete_action_size = discrete_action_size
        self.device = self.set_device()
        self.discount = d_r
        
        self.dnn_policy = DoubleDeepQNetwork.DNN(
            input_size=observation_dim_size, 
            output_size=discrete_action_size
            ).to(self.device)
        self.dnn_target = DoubleDeepQNetwork.DNN(
            input_size=observation_dim_size, 
            output_size=discrete_action_size
            ).to(self.device)

        self.optim = optim.Adam(self.dnn_policy.parameters(), lr=l_r)

        self.dnn_update_counter = 0
        self.loss_history = [] # a List of floats
    
    def set_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        return device
    
    def __call__(self, state):
        return self.get_optimal_action(state)
        
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        if type(state) == type(np.array([0])):
            state_tensor = torch.from_numpy(state)
        elif type(state) != type(torch.tensor([0])):
            raise Exception("State passed to get_optimal_action should be a np.array or torch.tensor.")        
        state_tensor = state_tensor.to(self.device).unsqueeze(0)
        prediction = self.dnn_policy(state_tensor)
        return torch.argmax(prediction).numpy()
    
    def save(self, taskName, path=None):
        creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if path is None: 
            path = f"{DNN_SAVE_FOLDER}/{taskName}_{creation_time}.pth"
        torch.save(self.dnn_policy.state_dict(), path)

    def load(self, path):
        self.dnn_policy.load_state_dict(torch.load(path))
        # move to self.device
        self.dnn_policy.to(self.device)
        
    # Updates the algorithm at the end of episode
    def update(self, buffer : NdArrayBuffer):
        """
        Updates the DDQN algorithms based on the given buffer.
        Returns the loss obtained in this update loop.

        :param NdArrayBuffer buffer: The buffer we use to update the algorithm.
        :return float loss: Returns the loss within this update.
        """
        BATCH_SIZE = 32
        
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
            
    # Updates the target dnn by setting its values equal to that of the policy dnn
    def _update_target(self):
        # do hard update
        if self.hard_update_every_N_updates != None:
            self.dnn_update_counter += 1
            if self.dnn_update_counter % self.hard_update_every_N_updates == 0:
                self.dnn_target.load_state_dict(self.dnn_policy.state_dict())
        # or soft update
        elif self.soft_update_coefficient != None:
            for param, target_param in zip(self.dnn_policy.parameters(), self.dnn_target.parameters()):
                target_param.data.copy_(self.soft_update_coefficient * 
                                        param.data + 
                                        (1 - self.soft_update_coefficient) * 
                                        target_param.data)
    
    def show_loss_history(self, task_name):
        plt.clf()
        plt.plot(range(0, len(self.loss_history)), self.loss_history)
        plt.savefig(f"{task_name}_loss_history_fig.png")
        plt.show()

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