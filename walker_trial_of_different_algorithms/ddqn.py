"""
A DDQN algorithm to be used as learning algorithm.
I want to try adding a component which predicts good actions to be evaluated using the 
Q learning approach, then takes the best action.

This will be another DNN which takes a state and spits out about 100 action pairs
that are 
"""

from datetime import datetime
import random
from typing import NamedTuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DNN_SAVE_FOLDER = "./dnns"

UPDATE_TARGET_DNN_EVERY_N = 4096
MAX_BUFFER_SIZE = 4096

ACTION_DISCRETIZED_NUMBER = 100
MAX, MIN = 0.5, -0.5

class Experience(NamedTuple):
    """
    A single transition experience in the given world, consisting of:
    - observation : an np.ndarray*
    - action : often either a name string, or an int
    - reward : float*
    - done flag : bool
    - next observation : an np.ndarray*
    *The types for observations and rewards are given as indicated in:
    https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#mlagents_envs.base_env
    """
    obs : np.ndarray
    action : np.ndarray
    reward : float
    done : bool
    next_obs : np.ndarray

Buffer = List[Experience]

def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
    None

    Returns:
    Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Code executes on CPU.")
    else:
        print("Code executes on GPU.")

    return device


class DDQN:
    DEVICE = set_device()

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
    

    def __init__(self, observation_size, action_size, l_r=0.1, d_r=0.95):
        self.discount = d_r

        self.observation_size = observation_size
        self.action_size = action_size
        
        dnn = DDQN.DNN
        self.dnn_policy = dnn(
            input_size=observation_size, 
            output_size=action_size * ACTION_DISCRETIZED_NUMBER
            ).to(DDQN.DEVICE)
        self.dnn_target = dnn(
            input_size=observation_size, 
            output_size=action_size * ACTION_DISCRETIZED_NUMBER
            ).to(DDQN.DEVICE)

        self.optim = optim.Adam(self.dnn_policy.parameters(), lr=l_r)
    
    def __call__(self, state_numpy):
        return self.dnn_policy(torch.from_numpy(state_numpy))
        
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        state_tensor = torch.tensor(state).to(DDQN.DEVICE).unsqueeze(0)
        prediction = self.dnn_policy(state_tensor)
        return torch.argmax(prediction).item()
    
    def save(self, taskName, path=None):
        creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if path is None: 
            path = DNN_SAVE_FOLDER + "/" + taskName + creation_time + ".pth"
        torch.save(self.dnn_policy.state_dict(), path)

    def load(self, taskName, path=None):
        if path is None: path = "./dnns/" + taskName + ".pth"
        self.dnn_policy.load_state_dict(torch.load(path))
        # move to DEVICE
        self.dnn_policy.to(DDQN.DEVICE)
        
    # Updates the algorithm at the end of episode
    def update(self, buffer : Buffer):
        BATCH_SIZE = 32
        EPOCHS = 12
        
        if len(buffer) > MAX_BUFFER_SIZE:
            buffer.reverse() #ensures that newest experience at end is kept
            buffer = buffer[:MAX_BUFFER_SIZE]
        elif len(buffer) < BATCH_SIZE: #if buffer too small, pass
            return
        
        loss_fn = nn.MSELoss()
        reshape_size = (len(batch), self.action_size, ACTION_DISCRETIZED_NUMBER)

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