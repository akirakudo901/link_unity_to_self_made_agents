"""
Visual Q network which takes in a visual input of the world, and outputs the value of 
each action in this world.
"""

from datetime import datetime
from math import floor
import random
from typing import Tuple, List

import numpy as np
import torch
from torch.nn import Parameter

from gridworld_base_trainer import GridWorldExperience, Buffer

DEFAULT_Q_FOLDER = "./gridworld_example_breakdown/q_networks"


"""
### Export PyTorch Model to ONNX

The following cell provides an example of some of the extra tensors a model needs to work for ML-Agents inference with Barracuda. The GridWorldColab scene is configured to work with this ONNX file.
Only policy models need to be exported for inference and they need the following additional tensors:

*   All models need version_number
*   All models need memory_size
*   Models with continuous outputs need continuous_action_output_shape
*   Models with discrete outputs need discrete_action_output_shape and an additional mask input that matches the shape of the discrete outputs
*   The mask input must be connected to the outputs or it will be pruned on export, if mask values aren't being set they will be 1, so multiplying the discrete outputs by the mask will have no effect

"""


class WrapperNet(torch.nn.Module):
    def __init__(
            self,
            qnet,
            discrete_output_sizes: List[int],
    ):
        """
        Wraps the VisualQNetwork adding extra constants and dummy mask inputs
        required by runtime inference with Barracuda.

        For environment continuous actions outputs would need to add them
        similarly to how discrete action outputs work, both in the wrapper
        and in the ONNX output_names / dynamic_axes.
        """
        super(WrapperNet, self).__init__()
        self.qnet = qnet

        # version_number
        #   MLAgents1_0 = 2   (not covered by this example)
        #   MLAgents2_0 = 3
        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        # memory_size
        # TODO: document case where memory is not zero.
        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        # discrete_action_output_shape
        output_shape = torch.Tensor([discrete_output_sizes])
        self.discrete_shape = Parameter(output_shape, requires_grad=False)

    # if you have discrete actions ML-agents expects corresponding a mask
    # tensor with the same shape to exist as input

    def forward(self, visual_obs: torch.tensor, mask: torch.tensor):
        qnet_result = self.qnet(visual_obs)
        # Connect mask to keep it from getting pruned
        # Mask values will be 1 if you never call SetActionMask() in
        # WriteDiscreteActionMask()
        qnet_result = torch.mul(qnet_result, mask)
        action = torch.argmax(qnet_result, dim=1, keepdim=True)
        return [action], self.discrete_shape, self.version_number, self.memory_size


class VisualQNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        encoding_size: int,
        output_size: int
    ):
        """
        Creates a neural network that takes as input a batch of images (3
        dimensional tensors) and outputs a batch of outputs (1 dimensional
        tensors)
        """
        super(VisualQNetwork, self).__init__()
        height = input_shape[1]
        width = input_shape[2]
        initial_channels = input_shape[0]
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = torch.nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = torch.nn.Conv2d(16, 32, [4, 4], [2, 2])
        self.dense1 = torch.nn.Linear(self.final_flat, encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size, output_size)

        # stores the number of actions as output size
        self.num_actions = output_size

    def forward(self, visual_obs: torch.tensor):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
        hidden = torch.relu(hidden)
        hidden = self.dense2(hidden)
        return hidden

    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w

    # Saves the model
    def save_model(
        self,
        example_obs: GridWorldExperience,
        path: str = None
    ):
        if path is not None:
            export_path = path
        else:
            export_path = DEFAULT_Q_FOLDER + "/" + "GridWorldColab_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M") + ".onnx"

        torch.onnx.export(
            WrapperNet(self, [self.num_actions]),
            # A tuple with an example of the input tensors
            (torch.tensor([example_obs]), torch.ones(1, self.num_actions)),
            export_path,
            opset_version=9,
            # input_names must correspond to the WrapperNet forward parameters
            # obs will be obs_0, obs_1, etc.
            input_names=["obs_0", "action_masks"],
            # output_names must correspond to the return tuple of the WrapperNet
            # forward function.
            output_names=["discrete_actions", "discrete_action_output_shape",
                          "version_number", "memory_size"],
            # All inputs and outputs should have their 0th dimension be designated
            # as 'batch'
            dynamic_axes={'obs_0': {0: 'batch'},
                          'action_masks': {0: 'batch'},
                          'discrete_actions': {0: 'batch'},
                          'discrete_action_output_shape': {0: 'batch'}
                          }
        )

    # Updates the q net with relevant data
    def update_q_net(
        self,
        optimizer: torch.optim,
        buffer: Buffer,
        action_size: int
    ):
        """
        Performs an update of the Q-Network using the provided optimizer and buffer
        """
        BATCH_SIZE = 1000
        NUM_EPOCH = 3
        GAMMA = 0.9
        batch_size = min(len(buffer), BATCH_SIZE)
        random.shuffle(buffer)
        # Split the buffer into batches
        batches = [
            buffer[batch_size * start: batch_size * (start + 1)]
            for start in range(int(len(buffer) / batch_size))
        ]
        for _ in range(NUM_EPOCH):
            for batch in batches:
                # Create the Tensors that will be fed in the network
                obs = torch.from_numpy(np.stack([ex.obs for ex in batch]))
                reward = torch.from_numpy(
                    np.array([ex.reward for ex in batch],
                             dtype=np.float32).reshape(-1, 1)
                )
                done = torch.from_numpy(
                    np.array([ex.done for ex in batch],
                             dtype=np.float32).reshape(-1, 1)
                )
                action = torch.from_numpy(
                    np.stack([ex.action for ex in batch]))
                next_obs = torch.from_numpy(
                    np.stack([ex.next_obs for ex in batch]))

                # Use the Bellman equation to update the Q-Network
                target = (
                    reward
                    + (1.0 - done)
                    * GAMMA
                    * torch.max(self(next_obs).detach(), dim=1, keepdim=True).values
                )
                mask = torch.zeros((len(batch), action_size))
                mask.scatter_(1, action, 1)
                prediction = torch.sum(self(obs) * mask, dim=1, keepdim=True)
                criterion = torch.nn.MSELoss()
                loss = criterion(prediction, target)

                # Perform the backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
