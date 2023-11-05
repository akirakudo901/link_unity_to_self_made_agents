"""
This is an implementation of SAC by Zhihan Yang found here: 
[https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py]

I downloaded this to see if this implementation works well with my environment - 
if it does, there lies a problem in my SAC implementation.
If it doesn't, there lies an issue in the gym base trainer.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from typing import NamedTuple

from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm
from models.trainers.utils.buffer import NdArrayBuffer

class Batch(NamedTuple):
    s : torch.tensor #state
    a : torch.tensor #action
    r : torch.tensor #reward
    d : torch.tensor #done
    ns : torch.tensor #next state


def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=5,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class NormalPolicyNet(nn.Module):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self, input_dim, action_dim, action_ranges):
        super(NormalPolicyNet, self).__init__()

        self.action_avgs       = torch.tensor([(range[1] + range[0]) / 2 for range in action_ranges])
        self.action_multiplier = torch.tensor([(range[1] - range[0]) / 2 for range in action_ranges])

        self.shared_net   = get_net(num_in=input_dim, num_out=64, final_activation=nn.ReLU())
        self.means_net    = nn.Linear(64, action_dim)
        self.log_stds_net = nn.Linear(64, action_dim)

    def forward(self, states: torch.tensor):

        out = self.shared_net(states)
        means, log_stds = self.means_net(out), self.log_stds_net(out)

        # the gradient of computing log_stds first and then using torch.exp
        # is much more well-behaved then computing stds directly using nn.Softplus()
        # ref: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L26

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))

        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

class QNet(nn.Module):

    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self, input_dim, action_dim):
        super(QNet, self).__init__()
        self.net = get_net(num_in=input_dim+action_dim, num_out=1, final_activation=None)

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=1))

class ZhihanSoftActroCritic(PolicyLearningAlgorithm):

    ALGORITHM_NAME = "ZhihanSAC"

    def __init__(self, input_dim, action_dim, env):
        super().__init__(env=env)
        self.policy   = NormalPolicyNet(input_dim=input_dim, action_dim=action_dim, action_ranges=self.act_ranges)
        self.Normal   = NormalPolicyNet(input_dim=input_dim, action_dim=action_dim, action_ranges=self.act_ranges)
        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=1e-3)

        self.Q1       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-3)

        self.Q2       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.alpha = 0.1
        self.polyak = 0.995

        #AKIRA ADDITION
        self.qnet1_loss_history = []
        self.qnet2_loss_history = []
        self.policy_loss_history = []

    # ==================================================================================================================
    # Helper methods (it is generally not my style of using helper methods but here they improve readability)
    # ==================================================================================================================

    def min_i_12(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        return torch.min(a, b)

    def sample_action_and_compute_log_pi(self, state: torch.tensor, use_reparametrization_trick: bool) -> tuple:
        mu_given_s = self.Normal(state)  # in paper, mu represents the normal distribution
        # in paper, u represents the un-squashed action; nu stands for next u's
        # actually, we can just use reparametrization trick in both Step 12 and 14, but it might be good to separate
        # the two cases for pedagogical purposes, i.e., using reparametrization trick is a must in Step 14
        u = mu_given_s.rsample() if use_reparametrization_trick else mu_given_s.sample()
        a = torch.tanh(u)
        # the following line of code is not numerically stable:
        # log_pi_a_given_s = mu_given_s.log_prob(u) - torch.sum(torch.log(1 - torch.tanh(u) ** 2), dim=1)
        # ref: https://github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
        return a, log_pi_a_given_s

    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    # ==================================================================================================================
    # Methods for learning
    # ==================================================================================================================

    def update(self, experiences : NdArrayBuffer, seed : int=None) -> None:

        # ========================================
        # Step 12: calculating targets
        # ========================================

        # sample the experiences from the buffer
        batch_size = 64

        if experiences.size() < batch_size: return

        observations, actions, rewards, dones, next_observations = experiences.sample_random_experiences(
            num_samples=batch_size
        )
        b = Batch(s=torch.from_numpy(observations).view(batch_size, -1), 
                  a=torch.from_numpy(actions).view(batch_size, -1), 
                  r=torch.from_numpy(rewards).view(batch_size, -1), 
                  d=torch.from_numpy(dones).view(batch_size, -1), 
                  ns=torch.from_numpy(next_observations).view(batch_size, -1))

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_and_compute_log_pi(b.ns, use_reparametrization_trick=False)
            targets = b.r + self.gamma * (1 - b.d) * \
                      (self.min_i_12(self.Q1_targ(b.ns, na), self.Q2_targ(b.ns, na)) - self.alpha * log_pi_na_given_ns)

        # ========================================
        # Step 13: learning the Q functions
        # ========================================

        Q1_predictions = self.Q1(b.s, b.a)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)
        self.qnet1_loss_history.append(Q1_loss.item()) #AKIRA ADDED

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        Q2_predictions = self.Q2(b.s, b.a)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)
        self.qnet2_loss_history.append(Q2_loss.item()) #AKIRA ADDED

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False

        a, log_pi_a_given_s = self.sample_action_and_compute_log_pi(b.s, use_reparametrization_trick=True)
        policy_loss = - torch.mean(self.min_i_12(self.Q1(b.s, a), self.Q2(b.s, a)) - self.alpha * log_pi_a_given_s)
        self.policy_loss_history.append(policy_loss.item()) #AKIRA ADDED

        self.Normal_optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.Normal)
        self.Normal_optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True

        # ========================================
        # Step 15: update target networks
        # ========================================

        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    def act(self, state: np.array) -> np.array:
        state = torch.tensor(state).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(state, use_reparametrization_trick=False)
        return action.numpy()[0]  # no need to detach first because we are not using the reparametrization trick

    def save_actor(self, save_dir: str, filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.Normal.state_dict(), os.path.join(save_dir, filename))

    def load_actor(self, save_dir: str, filename: str) -> None:
        self.Normal.load_state_dict(torch.load(os.path.join(save_dir, filename)))
    
    
################################################
# AKIRA CODE: OVERWRITE AREA
################################################

    def get_optimal_action(self, state):
        """
        Given the state, returns the corresponding optimal action under current knowledge.

        :param torch.tensor or np.ndarray state: The state for which we return the optimal action.
        """
        return self.act(state)

    def save(self, task_name : str, save_dir : str):
        """
        Saves the current policy.

        :param str task_name: The name of the task according to which we save the algorithm.
        :param str save_dir: The directory to which this policy is saved.
        """
        pass
    
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

    def load(self, path : str):
        """
        Loads the current policy.

        :param str path: Path from which we load the algorithm.
        """
        pass
    
    def _load_parameter_dict(self, dict):
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
        total_loss_history = [(self.qnet1_loss_history[i] + 
                            self.qnet2_loss_history[i] + 
                            self.policy_loss_history[i])
                            for i in range(min(
                                len(self.qnet1_loss_history),
                                len(self.qnet2_loss_history),
                                len(self.policy_loss_history))
                                )]
        
        for h, src in [(self.qnet1_loss_history,  "qnet1"), 
                      (self.qnet2_loss_history,  "qnet2"),
                      (self.policy_loss_history, "policy_net"),
                      (total_loss_history,       "total")]:
            PolicyLearningAlgorithm.plot_history_and_save(history=h, loss_source=src, 
                                                             task_name=task_name, 
                                                             save_figure=save_figure, 
                                                             save_dir=save_dir)

    def _delete_saved_algorithms(self, dir : str, task_name : str, training_id : int):
        """
        Deletes the saved algorithm files.
        
        :param str dir: String specifying the path in which the deleted data is held.
        :param str task_name: Name specifying the type of task.
        :param int training_id: An integer specifying training id.
        """
        pass