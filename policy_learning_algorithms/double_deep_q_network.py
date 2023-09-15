"""
A DDQN algorithm to be used as learning algorithm.
"""

from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trainers.gym_base_trainer import NdArrayBuffer

DNN_SAVE_FOLDER = "trained_algorithms/DNN"

UPDATE_TARGET_DNN_EVERY_N = 4096

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
    

    def __init__(self, observation_size, action_size, device, l_r=0.1, d_r=0.95):
        self.discount = d_r

        self.obs_size = observation_size
        self.act_size = action_size
        self.device = device
        
        self.dnn_policy = DoubleDeepQNetwork.DNN(
            input_size=observation_size, 
            output_size=action_size
            ).to(self.device)
        self.dnn_target = DoubleDeepQNetwork.DNN(
            input_size=observation_size, 
            output_size=action_size
            ).to(self.device)

        self.optim = optim.Adam(self.dnn_policy.parameters(), lr=l_r)
    
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
        BATCH_SIZE = 32
        
        loss_fn = nn.MSELoss()

        # sample a minibatch of transitions from the buffer
        np_obs, np_act, np_rew, np_don, np_next_obs = buffer.sample_random_experiences(num_samples=BATCH_SIZE) #TODO no seed, how to incorporate it?
        states      = torch.from_numpy(np_obs)
        actions     = torch.from_numpy(np_act)
        rewards     = torch.from_numpy(np_rew)
        dones       = torch.from_numpy(np_don)
        next_states = torch.from_numpy(np_next_obs)
        # get the corresponding updated q val
        updated_q = (rewards + 
                     (1 - dones)
                     self.discount * 
                     torch.max(self.dnn_target(next_states).detach(), dim=1).values)
        # from action, make a mask 
        mask = torch.zeros(len(states), ACTION_NUM)
        mask.scatter_(1, actions.unsqueeze(1), 1)
        # apply mask to obtain the relevant predictions for current states
        compared_q = torch.sum(self.dnn_policy(states) * mask, dim=1)
        # calculate loss)
        loss = loss_fn(compared_q, updated_q)
        # propagate the result
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # update the target dnn appropriately after one update
        self._update_target()
            
    # Updates the target dnn by setting its values equal to that of the policy dnn
    def _update_target(self):
        self.dnn_target.load_state_dict(self.dnn_policy.state_dict())