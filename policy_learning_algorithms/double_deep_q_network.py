"""
A DDQN algorithm to be used as learning algorithm.
"""

from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trainers.gym_base_trainer import ListBuffer

DNN_SAVE_FOLDER = "trained_algorithms/DNN"

UPDATE_TARGET_DNN_EVERY_N = 4096

class DDQN:

    class DNN(nn.Module):

        def __init__(self, input_size, output_size):
            # Initializes a new DNN.
            super(DDQN.DNN, self).__init__()
            
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

        self.observation_size = observation_size
        self.action_size = action_size
        self.device = device
        
        self.dnn_policy = DDQN.DNN(
            input_size=observation_size, 
            output_size=action_size
            ).to(self.device)
        self.dnn_target = DDQN.DNN(
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
        return torch.argmax(prediction).item()
    
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
    def update(self, buffer : ListBuffer):
        BATCH_SIZE = 32
        EPOCHS = 12
        
        loss_fn = nn.MSELoss()
        reshape_size = (len(batch), self.action_size)

        for _ in range(EPOCHS):
            random.shuffle(buffer)
            batches = [
                buffer[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                for i in range(
                min(8, len(buffer) // BATCH_SIZE)
                )
            ]

            for batch in batches:
                # stack each entry into torch tensors to do further computation
                current_states = torch.from_numpy(np.stack([exp.obs for exp in batch]))
                continuous_actions = torch.from_numpy(np.stack([exp.action for exp in batch]))
                rewards = torch.from_numpy(np.stack([exp.reward for exp in batch]))
                next_states = torch.from_numpy(np.stack([exp.next_obs for exp in batch]))
                # get the corresponding updated q val
                next_state_rewards = torch.reshape(self.dnn_target(next_states).detach(), reshape_size)
                print(next_state_rewards.shape)
                updated_q = (rewards  #(32, 1)
                             + self.discount
                             * torch.max(next_state_rewards, dim=2, keepdim=True).values) #(32, 39)
                print(updated_q.shape)
                # from action, make a mask 
                mask = torch.zeros(reshape_size)
                discrete_actions = DDQN.convert_continuous_action_to_discrete(continuous_actions)
                print(discrete_actions.shape)
                print(mask.shape)
                mask.scatter_(2, discrete_actions.unsqueeze(2).type(torch.int64), 1)
                print(mask.shape)
                # apply mask to obtain the relevant predictions for current states
                compared_q = torch.sum(
                    torch.reshape(self.dnn_policy(current_states), reshape_size)
                    * mask, dim=2)
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