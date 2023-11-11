"""
A trial implementation of Soft-Actor Critic as documented in the spinning-up page:
https://spinningup.openai.com/en/latest/algorithms/sac.html
"""

import os
import math
import random
from typing import Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import wandb

from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm
from models.trainers.utils.buffer import Buffer

class SoftActorCritic(PolicyLearningAlgorithm):
    ALGORITHM_NAME = "SAC"
    NUMBER_DTYPE = torch.float32
    
    class QNet(nn.Module):
        """
        A neural network approximating the q-value function.
        """
        
        def __init__(self, observation_size : int, action_size : int, 
                     qnet_layer_sizes : Tuple[int] = (64, 64)):
            """
            Initializes one of the Q-function approximating NN which evaluates
            the value of any state-action pair. The input & output sizes for 
            hidden layers can be determined through qnet_layer_sizes.
            
            In the original SAC paper found at this page[https://arxiv.org/pdf/1801.01290.pdf],
            the following are set as found in D. hyperparameters:
            - number of hidden layers = 2
            - number of hidden units per layer = 256
            - nonlinearity = ReLU

            :param int observation_size: The size of the observation input vector.
            :param int action_size: The size of the action vectors.
            :param Tuple[int] qnet_layer_sizes: The size of interm layers creating the net. \
            Defaults to (64, 64).
            """
            super(SoftActorCritic.QNet, self).__init__()

            self.linear_relu_stack = PolicyLearningAlgorithm.create_net(
                input_size=observation_size + action_size,
                output_size=1,
                interim_layer_sizes=qnet_layer_sizes
                )

        
        def forward(self, obs : torch.tensor, actions : torch.tensor):
            """
            Passes obs and actions vector (or batches of them) into the nn to 
            return a value indicating each state-action pairs' q-value.

            :param torch.tensor obs: Batch of state vectors in the form [batch, obs_dim].
            :param torch.tensor actions: Batch of action vectors in the form [batch, act_dim].
            :return torch.tensor out: Q-values for each batch of state-action pair. \
                In the form [batch].
            """
            # concatenate input skipping batch dimension to input to nn
            inp = torch.cat((obs, actions), dim=1)
            # passing it through the NN
            x = self.linear_relu_stack(inp)
            out = torch.squeeze(x, dim=1)
            return out

    class Policy(nn.Module):
        """
        A neural network approximating the policy.
        Composed of a stochastic noise component and a deterministic component
        which maps the noises to corresponding actions - which is required to apply the 
        reparameterization trick, as proposed in the original paper.
        """

        def __init__(self, observation_size : int, action_size : int, 
                     action_ranges : Tuple[Tuple[float]],
                     policy_layer_sizes : Tuple[int] = (64, 64)):
            """
            Implements the neural network which maps observations to two vectors
            determining the corresponding distribution for the policy:
            - mean vector of approximated function
            - variance vector of approximated function (the covariance matrix of this 
            multivariate normal distribution is diagonal)

            In the original SAC paper found at this page[https://arxiv.org/pdf/1801.01290.pdf],
            the following are set as found in D. hyperparameters:
            - number of hidden layers = 2
            - numbre of hidden units per layer = 256
            - nonlinearity = ReLU

            :param int observation_size: The size of the observation input vector.
            :param int action_size: The size of the action vectors.
            :param Tuple[Tuple[int]] action_ranges: The high and low ranges of possible
            values in the action space, as ith value being (low, high).
            :param Tuple[int] policy_layer_sizes: The size of interm layers creating the net. \
            Defaults to (64, 64).
            """
            super(SoftActorCritic.Policy, self).__init__()
            # of shape [act_dim]
            self.action_avgs       = torch.tensor([(range[1] + range[0]) / 2 for range in action_ranges]).to(SoftActorCritic.NUMBER_DTYPE)
            # of shape [act_dim]
            self.action_multiplier = torch.tensor([(range[1] - range[0]) / 2 for range in action_ranges]).to(SoftActorCritic.NUMBER_DTYPE)
            
            self.linear_relu_stack = PolicyLearningAlgorithm.create_net(
                input_size=observation_size,
                output_size=1,
                interim_layer_sizes=policy_layer_sizes
            )[:-1]

            self.mean_layer = nn.Linear(policy_layer_sizes[-1], action_size) #initial implementation
            self.sd_layer = nn.Linear(policy_layer_sizes[-1], action_size)

        def forward(self,  #new
                    obs : torch.tensor,
                    num_samples : int = 1,
                    deterministic : bool = False
                    ):
            """
            Takes in the observation to spit out num_samples actions sampled from the 
            policy, as well as their log probabilities.
            The policy is approximated by a gaussian with mean/sd obtained by evaluating
            a neural network at observation.
            
            :param torch.tensor obs: The observations at which the policy is evaluated. \
            Of the form [batch, obs_dim].
            :param int num_samples: The number of samples we obtain for each observation.
            :param bool deterministic: Whether the return should be deterministic (only at inference)

            if deterministic is True:
            :return torch.tensor: Actions obtained as means of the gaussian distributions which \
            are used as estimates of the policy at the corresponding states. The result is squashed \
            and adjusted to match the action_ranges values. Of shape, [batch, act_dim].

            if deterministic is False:
            :return torch.tensor squashed: The actions sampled from the policy and squashed between \
            -1 and 1 using the tanh (as done with the original paper to limit possible action range) \
            before being adjusted to match the action_ranges values. Of shape, [batch, act_dim].
            :return torch.tensor log_probs: The log probability for corresponding actions in squashed \
            after appropriate adjustment. Of shape, [batch].
            """
            
            # we squash action values to be between a given bounary using tanh and adjustment
            def squashing_function(actions : torch.tensor):
                """
                Squashes the input action using tanh then adjusts it 
                based on the mean and range of actions.

                :param torch.tensor actions: The action to be adjusted. Of the form [batch, act_dim].
                :return: The adjusted actions. Of the form [batch, act_dim].
                """
                squashed_neg_one_to_one = torch.tanh(actions)
                return (squashed_neg_one_to_one * 
                        self.action_multiplier.to(squashed_neg_one_to_one.device).detach() + 
                        self.action_avgs.to(squashed_neg_one_to_one.device).detach())

            # we obtain the mean myu and sd sigma of the gaussian distribution
            stack_out = self.linear_relu_stack(obs) #[batch, stack_out_size]
            myus = self.mean_layer(stack_out) #[batch, act_dim]

            # # ONE WAY TO DO IT (FROM SPINNING UP?)
            # LOG_STD_MIN, LOG_STD_MAX = -5, 2
            # log_std = torch.tanh(self.sd_layer(stack_out))
            # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            # sigmas = torch.exp(log_std)
            # ONE WAY TO DO IT END

            # ANOTHER WAY TO DO IT (FROM https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py)
            LOG_STD_MAX = 2
            LOG_STD_MIN = -20

            log_stds = self.sd_layer(stack_out)
            sigmas = torch.exp(torch.clamp(log_stds, min=LOG_STD_MIN, max=LOG_STD_MAX)) #[batch, act_dim]
            #END EXPERIMENTAL
                        
            # if deterministic (while in inference), return the mean of distributions
            # corresponding to actions at time of inference, but squashed as needed
            if deterministic: return squashing_function(myus) #[batch, act_dim]
            
            # then evaluate the probability that action is chosen under the distribution
            dist = Independent(Normal(loc=myus, scale=sigmas), #same as MultivariateNormal w/ diagonal covariance
                               reinterpreted_batch_ndims=1)
            actions = dist.rsample().to(SoftActorCritic.NUMBER_DTYPE) #[batch, act_dim]
            squashed = squashing_function(actions) #[batch, act_dim]
       
            # pure_probabilities is log_prob for each action when sampled from each normal distribution
            pure_log_probabilities = dist.log_prob(actions).to(SoftActorCritic.NUMBER_DTYPE) #[batch]
            
            def _correct_for_squash(log_probs, actions):
                """
                Correct for squashing.

                :param torch.tensor log_probs: Log probabilities. Of the form [batch].
                :param torch.tensor actions: Sampled actions. Of the form [batch, act_dim].
                :return: The corrected probability, of the form [batch].
                """
                multiplier = self.action_multiplier.to(actions.device)
                jacobian_trace = torch.sum(torch.log(multiplier) + 
                                           (2. * (math.log(2) - 
                                                  actions - 
                                                  torch.nn.functional.softplus(-2. * actions))), dim=1)
                return log_probs - jacobian_trace

            log_probs = _correct_for_squash(
                pure_log_probabilities, actions
                )            
                 
            return squashed, log_probs

        # def forward(self,  #old
        #             obs : torch.tensor,
        #             num_samples : int = 1,
        #             deterministic : bool = False
        #             ):
        #     """
        #     Takes in the observation to spit out num_samples actions sampled from the 
        #     policy, as well as their log probabilities.
        #     The policy is approximated by a gaussian with mean/sd obtained by evaluating
        #     a neural network at observation.
            
        #     *num_samples seems to be set to 1 in the original implementation from the first 
        #     SAC paper & in the Spinning Up implementation; I might not have a concrete
        #     understanding yet of why that is the choice (instead of, let's say, sampling
        #     100 actions and calculate them all for the given task).
        #     *I am also a bit confused as to why we simply compute the log probability of 
        #     the action at the time, rather than directly calculating the entropy of the
        #     gaussian, which we have at hand.

        #     :param torch.tensor obs: The observations at which the policy is evaluated.
        #     :param int num_samples: The number of samples we obtain for each observation.
        #     :param bool deterministic: Whether the return should be deterministic (only at inference)

        #     if deterministic is True:
        #     :return torch.tensor myus: Actions obtained as means of the gaussian distributions which
        #     are used as estimates of the policy at the corresponding states. The result is squashed 
        #     and adjusted to match the action_ranges values.

        #     if deterministic is False:
        #     :return torch.tensor squashed: The actions sampled from the policy and squashed between
        #     -1 and 1 using the tanh (as done with the original paper to limit possible action range)
        #     before being adjusted to match the action_ranges values.
        #     :return torch.tensor log_probs: The log probability for corresponding actions in squashed
        #     after appropriate adjustment.
        #     """
            
        #     # we squash action values to be between a given bounary using tanh and adjustment
        #     def squashing_function(actions : torch.tensor):
        #         squashed_neg_one_to_one = torch.tanh(actions)
        #         return (squashed_neg_one_to_one * 
        #                 self.action_multiplier.to(squashed_neg_one_to_one.device).detach() + 
        #                 self.action_avgs.to(squashed_neg_one_to_one.device).detach())

        #     # we obtain the mean myu and sd sigma of the gaussian distribution
        #     stack_out = self.linear_relu_stack(obs)            
        #     myus = self.mean_layer(stack_out)

        #     # # ONE WAY TO DO IT (FROM SPINNING UP?)
        #     # LOG_STD_MIN, LOG_STD_MAX = -5, 2
        #     # log_std = torch.tanh(self.sd_layer(stack_out))
        #     # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        #     # sigmas = torch.exp(log_std)
        #     # ONE WAY TO DO IT END

        #     # ANOTHER WAY TO DO IT (FROM https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py)
        #     LOG_STD_MAX = 2
        #     LOG_STD_MIN = -20

        #     sigmas = torch.exp(torch.clamp(self.sd_layer(stack_out), min=LOG_STD_MIN, max=LOG_STD_MAX))
        #     #END EXPERIMENTAL
                        
        #     # if deterministic (while in inference), return the mean of distributions
        #     # corresponding to actions at time of inference, but squashed as needed
        #     if deterministic: return squashing_function(myus)
            
        #     # then evaluate the probability that action is chosen under the distribution
        #     dist = Independent(Normal(loc=myus, scale=sigmas),
        #                        reinterpreted_batch_ndims=2) #MultivariateNormal with diagonal covariance
        #     actions_num_samples_first = dist.rsample(sample_shape=(num_samples, )).to(SoftActorCritic.NUMBER_DTYPE)
        #     actions = torch.transpose(actions_num_samples_first, dim0=0, dim1=1)
        #     squashed = squashing_function(actions)
       
        #     # pure_probabilities is log_prob for each action when sampled from each normal distribution
        #     # aggregating over the entire action, it becomes their sum (product of independent events but logged)
            
        #     before_correction = dist.log_prob(actions_num_samples_first).to(SoftActorCritic.NUMBER_DTYPE) #EXPERIMENTAL

        #     # pure_log_probabilities = torch.transpose(dist.log_prob(actions_num_samples_first).to(SoftActorCritic.NUMBER_DTYPE), dim0=0, dim1=1)#EXPERIMENTAL
        #     # before_correction = torch.sum(pure_log_probabilities, dim=2)#EXPERIMENTAL
            
        #     log_probs = self._correct_for_squash(
        #         before_correction, actions
        #         )
                 
        #     return squashed, log_probs
                
        def _correct_for_squash(self, before : torch.tensor, actions : torch.tensor):
            """
            Corrects for the log probabilities following squashing
            using tanh. 

            Given before is a log probability and the transformation is elementwise tanh,
            the adjustment is 
                [before - Sum( log(1 - tanh^2(u_i)) )] 
                where u_i is the ith element in a single action.
                This is since
                1) Correction involves multiplying the original probability with the determinant
                    of the Jacobian of the transformation.
                2) The Jacobian of elementwise tanh is a diagonal, with each entry being each 
                    element of each action transformed by the derivative of tanh, which is 
                    (1 - tanh^2(x)). We divide by this factor here as after results from tanh 
                    correction of before.
                3) Since we are dealing with log probabilities, multiplication turn to addition.

            Furthermore, as we scale the action by different factors after squashing with tanh 
            to adjust to the actual action ranges, we multiply the whole result by 
            self.action_multiplier before the sum.

            :param torch.tensor before: The log probabilities to be corrected.
            Will be of the form [batch_size, num_samples].
            :param torch.tensor actions: The actions which log probabilities are in question.
            Will be of the form [batch_size, num_samples, action_size].
            :return torch.tensor after: The log probabilities after being corrected.
            Will be of the form [batch_size, num_samples].
            """
            # compute the trace of the Jacobian of tanh squashing for each action
            multiplier = self.action_multiplier.to(actions.device)

            # ORIGINAL IMPLEMENTATION
            # jacobian_trace = torch.sum(torch.log(multiplier * (1 - torch.tanh(actions).pow(2)) + 1e-6), dim=2)
            
            # apparently below is numerically equivalent to above but more stable:
            # https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py#L73 
            # formula is: 2 * (log(2) - x - softplus(-2x)) + log(multiplier)
            jacobian_trace = torch.sum(torch.log(multiplier) + 
                                       (2. * (torch.log(torch.tensor(2.)) - 
                                             actions - 
                                             torch.nn.functional.softplus(-2. * actions))), dim=2)
            # EXPERIMENTAL END
            
            # subtract it from before to yield after
            after = before - jacobian_trace
            
            return after

    def __init__(self,
                 q_net_learning_rate : float,
                 policy_learning_rate : float, 
                 discount : float,
                 temperature : float,
                 qnet_update_smoothing_coefficient : float,
                 pol_eval_batch_size : int,
                 pol_imp_batch_size : int,
                 update_qnet_every_N_gradient_steps : int,
                 obs_dim_size : int = None,
                 act_dim_size : int = None,
                 act_ranges : Tuple[Tuple[float]] = None,
                 qnet_layer_sizes : Tuple[int] = (64, 64),
                 policy_layer_sizes : Tuple[int] = (64, 64),
                 env = None
                 ):
        if (obs_dim_size == None or act_dim_size == None or act_ranges == None) and env == None:
            raise Exception("Either all of obs_dim_size, act_dim_size and act_ranges, or env, " +
                             "should be given!")
        
        super().__init__(obs_dim_size=obs_dim_size, act_dim_size=act_dim_size, 
                         act_ranges=act_ranges, env=env)

        self.q_net_l_r = q_net_learning_rate
        self.pol_l_r = policy_learning_rate
        self.d_r = discount
        self.alpha = temperature
        self.tau = qnet_update_smoothing_coefficient
        self.pol_eval_batch_size = pol_eval_batch_size
        self.pol_imp_batch_size = pol_imp_batch_size
        self.update_qnet_every_N_gradient_steps = update_qnet_every_N_gradient_steps
        self.qnet_layer_sizes = qnet_layer_sizes
        self.policy_layer_sizes = policy_layer_sizes

        self.qnet_update_counter = 0

        # two q-functions are used to approximate values during training
        self.qnet1 = SoftActorCritic.QNet(self.obs_dim_size, self.act_dim_size, qnet_layer_sizes).to(self.device)
        self.qnet2 = SoftActorCritic.QNet(self.obs_dim_size, self.act_dim_size, qnet_layer_sizes).to(self.device)
        # two target networks are updated less (e.g. every 1000 steps) to compute targets for q-function update
        self.qnet1_tar = SoftActorCritic.QNet(self.obs_dim_size, self.act_dim_size, qnet_layer_sizes).to(self.device)
        self.qnet2_tar = SoftActorCritic.QNet(self.obs_dim_size, self.act_dim_size, qnet_layer_sizes).to(self.device)
        # transfer the weights to sync
        self._update_target_networks(hard_update=True)
        # set up the optimizers
        self.optim_qnet1 = optim.Adam(self.qnet1.parameters(), lr=self.q_net_l_r)
        self.optim_qnet2 = optim.Adam(self.qnet2.parameters(), lr=self.q_net_l_r)
        
        # initialize policy 
        self.policy = SoftActorCritic.Policy(
            observation_size=self.obs_dim_size, action_size=self.act_dim_size, action_ranges=self.act_ranges,
            policy_layer_sizes=policy_layer_sizes
            ).to(self.device)
        self.optim_policy = optim.Adam(self.policy.parameters(), lr=policy_learning_rate)

        self.qnet1_loss_history = [] #List of float
        self.qnet2_loss_history = [] #List of float
        self.policy_loss_history = [] #List of float
    

    # def update(self, experiences : Buffer, seed : int=None): #old
    #     # a single experience contained in "experiences" is of the form: 
    #     # (obs, action, reward, done, next_obs)
    #     random.seed(seed)

    #     def get_new_seed():
    #         """
    #         Calcualtes a new seed given an old seed using random.randint...
    #         If passed 'seed' is None, returns numbers totally randomly. 
    #         Not aware of any possible issues, but I might have to look into it.
    #         """
    #         return random.randint(0, 1000)

    #     NUM_EVAL_STEPS = 1

    #     POLICY_EVAL_NUM_EPOCHS = 1
    #     POL_EVAL_FRESH_ACTION_SAMPLE_SIZE = 1

    #     NUM_IMP_STEPS = 1

    #     POLICY_IMP_NUM_EPOCHS = 1
    #     POL_IMP_FRESH_ACTION_SAMPLE_SIZE = 1

    #     observations, actions, rewards, dones, next_observations = SoftActorCritic._sample_experiences(
    #         experiences=experiences, num_samples=POLICY_EVAL_NUM_EPOCHS * self.pol_eval_batch_size, 
    #         device=self.device, seed=get_new_seed()
    #         )
        
    #     # freshly sample new actions in the current policy for each next observations
    #     # for now, we will sample FRESH_ACTION_SAMPLE_SIZE
    #     fresh_action_samples, fresh_log_probs = self.policy(next_observations, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        
    #     # 1 - policy evaluation

    #     policy_evaluation_gradient_step_count = 0

    #     for _ in range(POLICY_EVAL_NUM_EPOCHS):
    #         for i in range(experiences.size() // self.pol_eval_batch_size + 1):
    #             batch_start = i*self.pol_eval_batch_size
    #             batch_end = min((i+1)*self.pol_eval_batch_size, experiences.size())

    #             batch_obs     = observations[batch_start : batch_end]
    #             batch_actions = actions[batch_start : batch_end]
    #             batch_rewards = rewards[batch_start : batch_end]
    #             batch_dones   = dones[batch_start : batch_end]
    #             batch_nextobs = next_observations[batch_start : batch_end]
        
    #             # batch_action_samples' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, action_size]
    #             batch_action_samples = fresh_action_samples[batch_start : batch_end]
    #             # batch_log_probs' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE]
    #             batch_log_probs = fresh_log_probs[batch_start : batch_end]
                
    #             # first compute target value for all experiences (terminal ones are only the rewards)
    #             targets = self._compute_qnet_target(batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs)
                
    #             # then compute prediction value for all experiences
    #             predictions1 = self.qnet1(obs=batch_obs, actions=batch_actions)
    #             predictions2 = self.qnet2(obs=batch_obs, actions=batch_actions)
                
    #             # finally take loss through MSELoss
    #             criterion = nn.MSELoss()
    #             loss1 = criterion(predictions1, targets)
    #             loss2 = criterion(predictions2, targets)

    #             self.qnet1_loss_history.append(loss1.item())
    #             self.qnet2_loss_history.append(loss2.item())
    #             # backpropagate that loss to update q_nets
    #             self.optim_qnet1.zero_grad()
    #             loss1.backward()
    #             self.optim_qnet1.step()

    #             self.optim_qnet2.zero_grad()
    #             loss2.backward()
    #             self.optim_qnet2.step()
                
    #             # increment the update counter and update target networks every N gradient steps
    #             self.qnet_update_counter += 1
    #             if self.qnet_update_counter % self.update_qnet_every_N_gradient_steps == 0:
    #                 self._update_target_networks(hard_update=False)                    
                
    #             # finally break out of loop if the number of gradient steps exceeds NUM_STEPS
    #             policy_evaluation_gradient_step_count += 1
    #             if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break
    #         # also break out of outer loop
    #         if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break

    #     # 2 - policy improvement
    #     # at this point, the q-network weights are adjusted to reflect the q-value of
    #     # the current policy. We just have to take a gradient step with respect to the  
    #     # distance between this q-value and the current policy

    #     # TODO WE MIGHT BE ABLE TO REUSE THE PREVIOUS EXPERIENCES?
    #     # observations, actions, rewards, dones, next_observations = SoftActorCritic._sample_experiences(
    #     #     experiences=experiences, num_samples=POLICY_IMP_NUM_EPOCHS * self.pol_imp_batch_size, 
    #     #     device=self.device, seed=get_new_seed()
    #     #     )
        
    #     # in order to estimate the gradient, we again sample some actions at this point in time.
    #     # TODO COULD WE USE ACTIONS SAMPLED BEFORE WHICH WERE USED FOR Q-NETWORK UPDATE? NOT SURE
    #     fresh_action_samples2, fresh_log_probs2 = self.policy(observations, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        
    #     policy_improvement_gradient_step_count = 0

    #     for _ in range(POLICY_IMP_NUM_EPOCHS):
    #         for i in range(experiences.size() // self.pol_imp_batch_size + 1):
    #             batch_start = i*self.pol_imp_batch_size
    #             batch_end = min((i+1)*self.pol_imp_batch_size, experiences.size())

    #             batch_obs = observations[batch_start : batch_end]
                
    #             # batch_action_samples' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, action_size]
    #             batch_action_samples = fresh_action_samples2[batch_start : batch_end]
    #             # batch_log_probs' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE]
    #             batch_log_probs = fresh_log_probs2[batch_start : batch_end]
                
    #             # Then, we compute the loss function: which relates the exponentiated distribution of
    #             # the q-value function with the current policy's distribution, through KL divergence

    #             target_qval = self._compute_policy_val(batch_obs, batch_action_samples)
    #             policy_val = torch.mean(batch_log_probs, dim=1)

    #             # BELOW IS MY ORIGINAL APPROACH
    #             # criterion = nn.KLDivLoss(reduction="batchmean")
    #             # loss = criterion(policy_val, target_exped_qval)

    #             # BELOW IS CLEAN RL APPROACH, WHICH SHOULD BE IDENTICAL BY DEFINITION?
    #             loss = ((self.alpha * policy_val) - target_qval).mean()
                
    #             self.policy_loss_history.append(loss.item())
                
    #             # then, we improve the policy by minimizing this loss 
    #             self.optim_policy.zero_grad()
    #             loss.backward()
    #             self.optim_policy.step()

    #             # finally increment the improvement gradient step count
    #             policy_improvement_gradient_step_count += 1
    #             if policy_improvement_gradient_step_count >= NUM_IMP_STEPS: break
    #         # also break out of outer loop
    #         if policy_improvement_gradient_step_count >= NUM_IMP_STEPS: break
    
    ####TODO EXPERIMENTAL ###
    def update(self, experiences : Buffer, seed : int=None): #new
        if experiences.size() < self.pol_eval_batch_size: return 

        random.seed(seed)

        def clip_gradient(net: nn.Module) -> None: #TODO
            for param in net.parameters():
                param.grad.data.clamp_(-1, 1)

        def get_new_seed():
            """
            Calcualtes a new seed given an old seed using random.randint...
            If passed 'seed' is None, returns numbers totally randomly. 
            Not aware of any possible issues, but I might have to look into it.
            """
            return random.randint(0, 1000)
        
        # sample the experiences from the buffer
        observations, actions, rewards, dones, next_observations = SoftActorCritic._sample_experiences(
            experiences=experiences, num_samples=self.pol_eval_batch_size, 
            device=self.device, seed=get_new_seed()
            )
        
        # compute target value for all experiences (terminal ones only relate to rewards)
        def _compute_qnet_target(rewards : np.ndarray, dones : np.ndarray, next_observations : np.ndarray):
            """
            Computes the Q-net targets.

            :param np.ndarray rewards: The rewards, of shape [batch].
            :param np.ndarray dones: Done flags, of shape [batch].
            :param np.ndarray next_observations: The "next" observations, of shape [batch, obs_dim].
            :return np.ndarray targets: The target values, of shape [batch]
            """
    
            with torch.no_grad():
                #TODO INCORPORATE SEED INTO THE SELF.POLICY CALLS 
                # sample actions and their log probabilities from the policy
                # [batch, act_dim], [batch]
                action_samples, log_probs = self.policy(next_observations, 1, deterministic=False)
        
                qnet1_tar_preds = self.qnet1_tar(next_observations, action_samples) #[batch]
                qnet2_tar_preds = self.qnet2_tar(next_observations, action_samples) #[batch]
                minimum = torch.minimum(qnet1_tar_preds, qnet2_tar_preds) #[batch]
                targets = (rewards + 
                           self.d_r *
                           (1.0 - dones) *
                           (minimum - self.alpha * log_probs))
            return targets
        
        # TODO Can change between compute and not compute
        targets = _compute_qnet_target(rewards, dones, next_observations) #[batch]
        
        # take loss through MSELoss
        criterion = nn.MSELoss()
        
        # finally compute prediction value for all experiences
        predictions1 = self.qnet1(obs=observations, actions=actions) #[batch]
        loss1 = criterion(predictions1, targets) #[batch]
        # backpropagate that loss through qnet1
        self.optim_qnet1.zero_grad()
        loss1.backward()
        clip_gradient(self.qnet1)
        self.optim_qnet1.step()
        self.qnet1_loss_history.append(loss1.item())
        if wandb.run is not None: wandb.log({"QNet1 Loss" : loss1.item()})

        predictions2 = self.qnet2(obs=observations, actions=actions) #[batch]
        loss2 = criterion(predictions2, targets) #[batch]
        # backpropagate that loss through qnet1
        self.optim_qnet2.zero_grad()
        loss2.backward()
        clip_gradient(self.qnet2)
        self.optim_qnet2.step()
        self.qnet2_loss_history.append(loss2.item())
        if wandb.run is not None: wandb.log({"QNet2 Loss" : loss2.item()})


        # increment the update counter and update target networks every N gradient steps
        self.qnet_update_counter += 1
        if self.qnet_update_counter % self.update_qnet_every_N_gradient_steps == 0:
            self._update_target_networks(hard_update=False)

        #######################################################################
        # Update the policy
        #######################################################################
        for param in self.qnet1.parameters():
            param.requires_grad = False
        for param in self.qnet2.parameters():
            param.requires_grad = False

        # [batch, act_dim], [batch]
        action_samples2, log_probs2 = self.policy(observations, 1, deterministic=False)
        
        def _compute_policy_val(batch_obs, batch_action_samples):
            """
            Computes the policy value used as target for update.

            :param np.ndarray batch_obs: The observations, of shape [batch, obs_dim].
            :param np.ndarray batch_action_samples: The action samples, of shape [batch, act_dim].
            :return: The target Q-values, of shape [batch].
            """
            # with torch.no_grad():
            # predicted q_val for each qnet of the shape [batch, num_samples]
            qnet1_val = self.qnet1(batch_obs, batch_action_samples) #[batch]
            qnet2_val = self.qnet2(batch_obs, batch_action_samples) #[batch]
            # the minimum of those predictions of the shape [batch, num_samples]
            qval_minimum = torch.minimum(qnet1_val, qnet2_val) #[batch]
            
            # previous implemenbtations using the KL loss similarly to specified in the paper would 
            # exponentiate the Q-value before averaging them - but this presumably led to NaN gradients
            # and thus was replaced, as other implementations do, with a simple mean loss and no exponentiation
            # example of such implementations: https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py
            return qval_minimum

        # Then, we compute the loss function: which relates the exponentiated distribution of
        # the q-value function with the current policy's distribution, through KL divergence
        target_qval = _compute_policy_val(observations, action_samples2) #[batch]
        loss = ((self.alpha * log_probs2) - target_qval).mean() #[batch]
        
        # then, we improve the policy by minimizing this loss 
        self.optim_policy.zero_grad()
        loss.backward()
        clip_gradient(self.policy)
        self.optim_policy.step()
        self.policy_loss_history.append(loss.item())
        if wandb.run is not None: 
            wandb.log({"Policy Loss" : loss.item()})
            # also log the total loss for this single update step
            wandb.log({"Total Loss" : loss1.item() + loss2.item() + loss.item()})

        for param in self.qnet1.parameters():
            param.requires_grad = True
        for param in self.qnet2.parameters():
            param.requires_grad = True

    # def _policy_evaluation_step(self,
    #                             observations : np.ndarray,
    #                             actions : np.ndarray,
    #                             rewards : np.ndarray,
    #                             dones : np.ndarray,
    #                             next_observations : np.ndarray,
    #                             action_sample_batch_size : int):
        
    #     def single_policy_eval_step(observations : np.ndarray, 
    #                                 actions : np.ndarray,
    #                                 rewards : np.ndarray, 
    #                                 dones : np.ndarray, 
    #                                 next_observations : np.ndarray):

    #         #TODO INCORPORATE SEED INTO THE SELF.POLICY CALLS 
    #         # sample actions and their log probabilities from the policy
    #         action_samples, log_probs = self.policy(next_observations, action_sample_batch_size, deterministic=False)
            
    #         # compute target value for all experiences (terminal ones only relate to rewards)
    #         targets = self._compute_qnet_target(rewards, dones, next_observations, action_samples, log_probs)
            
    #         # then compute prediction value for all experiences
    #         predictions1 = self.qnet1(obs=observations, actions=actions)    
    #         predictions2 = self.qnet2(obs=observations, actions=actions)
            
    #         # finally take loss through MSELoss
    #         criterion = nn.MSELoss()
    #         loss1 = criterion(predictions1, targets)
    #         loss2 = criterion(predictions2, targets)

    #         self.qnet1_loss_history.append(loss1.item())
    #         self.qnet2_loss_history.append(loss2.item())
    #         # backpropagate that loss to update q_nets
    #         self.optim_qnet1.zero_grad()
    #         loss1.backward()
    #         self.optim_qnet1.step()

    #         self.optim_qnet2.zero_grad()
    #         loss2.backward()
    #         self.optim_qnet2.step()

    #     policy_evaluation_gradient_step_count = 0

    #     for _ in range(num_epochs):
    #         for i in range(experiences.size() // self.pol_eval_batch_size + 1):
    #             batch_start = i*self.pol_eval_batch_size
    #             batch_end = min((i+1)*self.pol_eval_batch_size, experiences.size())

    #             # batch_obs     = observations[batch_start : batch_end]
    #             # batch_actions = actions[batch_start : batch_end]
    #             # batch_rewards = rewards[batch_start : batch_end]
    #             # batch_dones   = dones[batch_start : batch_end]
    #             # batch_nextobs = next_observations[batch_start : batch_end]
        
    #             # batch_action_samples' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, action_size]
    #             # batch_action_samples = fresh_action_samples[batch_start : batch_end]
    #             # batch_log_probs' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE]
    #             # batch_log_probs = fresh_log_probs[batch_start : batch_end]
                
                
                
    #             # increment the update counter and update target networks every N gradient steps
    #             self.qnet_update_counter += 1
    #             if self.qnet_update_counter % self.update_qnet_every_N_gradient_steps == 0:
    #                 self._update_target_networks(hard_update=False)                    
                
    #             # finally break out of loop if the number of gradient steps exceeds NUM_STEPS
    #             policy_evaluation_gradient_step_count += 1
    #             if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break
    #         # also break out of outer loop
    #         if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break
                
    def _compute_policy_val(self, batch_obs, batch_action_samples):
        # predicted q_val for each qnet of the shape [batch, num_samples]
        qnet1_val = torch.cat([
            self.qnet1(batch_obs, torch.squeeze(batch_action_samples[:, i, :], dim=0))
            for i in range(batch_action_samples.shape[1])
            ], dim=1)
        qnet2_val = torch.cat([
            self.qnet2(batch_obs, torch.squeeze(batch_action_samples[:, i, :], dim=0))
            for i in range(batch_action_samples.shape[1])
            ], dim=1)
        # the minimum of those predictions of the shape [batch, num_samples]
        qval_minimum = torch.minimum(qnet1_val, qnet2_val)
        # the mean of those predictions of the shape [batch]
        mean_exp_qval = torch.mean(qval_minimum, dim=1, dtype=SoftActorCritic.NUMBER_DTYPE)
      
        # previous implemenbtations using the KL loss similarly to specified in the paper would 
        # exponentiate the Q-value before averaging them - but this presumably led to NaN gradients
        # and thus was replaced, as other implementations do, with a simple mean loss and no exponentiation
        # example of such implementations: https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py
        return mean_exp_qval

    def _compute_qnet_target(self, batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs):
        with torch.no_grad():
            qnet1_tar_preds = torch.cat([
                self.qnet1_tar(batch_nextobs, torch.squeeze(batch_action_samples[:, i, :], dim=0))
                for i in range(batch_action_samples.shape[1])
                ], dim=1)
            qnet2_tar_preds = torch.cat([
                self.qnet2_tar(batch_nextobs, torch.squeeze(batch_action_samples[:, i, :], dim=0))
                for i in range(batch_action_samples.shape[1])
                ], dim=1)
            minimum = torch.minimum(qnet1_tar_preds, qnet2_tar_preds)
            mean_of_minimum = torch.mean(minimum, dim=1, dtype=SoftActorCritic.NUMBER_DTYPE)
            mean_of_log = torch.mean(batch_log_probs, dim=1)
            targets = (
                        batch_rewards + 
                        self.d_r *
                        (1.0 - batch_dones.to(SoftActorCritic.NUMBER_DTYPE)) *
                        (mean_of_minimum - self.alpha * mean_of_log)
                    )
           
        return targets
    
    def _update_target_networks(self, hard_update : bool = False):
        """
        Updates the target networks utilizing the target smoothing coefficient 
        to slowly adjust parameters by considering from the corresponding q-networks.
        """
        with torch.no_grad():
            if hard_update:
                self.qnet1_tar.load_state_dict(self.qnet1.state_dict())
                self.qnet2_tar.load_state_dict(self.qnet2.state_dict())
            else:
                for param, target_param in zip(self.qnet1.parameters(), self.qnet1_tar.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qnet2.parameters(), self.qnet2_tar.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _sample_experiences(experiences : Buffer, num_samples : int, device = None, seed : int = None):
        """
        Randomly samples num_samples experiences from the given experiences buffer, expanding them 
        into observations, actions, rewards, done flags, and next observations, to be returned.

        :param Buffer experiences: A Buffer containing obs, action, reward, done and next_obs.
        :param int num_samples: The number of random samples we get from the buffer.
        :param device: The device to which we move resulting tensors, if any.
        :param int seed: The seed used for random sampling of experiences, if specified.
        Not used if None.
        :return Tuple[torch.tensor]: Tensors of each component in the Buffer; \ 
        observations, actions, rewards, dones, next_observations, with \
        shapes [batch, obs_dim], [batch, act_dim], [batch], [batch], [batch, obs_dim].
        """
        np_obs, np_act, np_rew, np_don, np_next_obs = experiences.sample_random_experiences(num_samples=num_samples, seed=seed)
        observations, actions, rewards, dones, next_observations = (torch.from_numpy(np_obs).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_act).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_rew).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_don).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_next_obs).to(SoftActorCritic.NUMBER_DTYPE))
        if device != None:
            observations, actions, rewards, dones, next_observations = (observations.to(device),
                                                                        actions.to(device),
                                                                        rewards.to(device),
                                                                        dones.to(device),
                                                                        next_observations.to(device))

        return observations,actions,rewards,dones,next_observations

    def __call__(self, state : Union[torch.tensor, np.ndarray], deterministic=True):
        """
        A raw call to the function which then calls get_optimal_action. 
        Can choose whether we compute the action deterministically or stochastically.

        :param torch.tensor or np.ndarray state: The state of observation in question \
        to which this algorithm was applied.
        :param bool deterministic: Whether the action is computed deterministically or \
        stochastically.
        """
        return self.get_optimal_action(state, deterministic=deterministic)
    
    def get_optimal_action(self, state : Union[torch.tensor, np.ndarray], 
                           deterministic : bool=True):
        """
        Computes the currently optimal action given an observation state.
        State can be either a torch tensor or numpy array, both being converted
        into a torch tensor before further processing is done.
        Can choose between deterministic or stochastic computation.

        :param torch.tensor or np.ndarray state: The observation state. 
        :param bool deterministic: Whether to compute the action deterministically \
        (return the mean of the distribution) or stochastically (sample from the distribution).
        :return np.ndarray action: The action the policy deems optimal as ndarray. 
        """
        with torch.no_grad():
            state_tensor = super().get_optimal_action(state)
            if deterministic:
                action = self.policy(obs=state_tensor.to(self.device), deterministic=deterministic)
            else: 
                action, _ = self.policy(obs=state_tensor.to(self.device), deterministic=deterministic)
            return action.detach().cpu().numpy()
    
    def save(self, task_name: str, save_dir : str = None):
        """
        Saves the current policy.

        :param str task_name: The name of the task according to which we save the algorithm.
        :param str save_dir: The directory to which this policy is saved.
        """
        save_dir = PolicyLearningAlgorithm.get_saving_directory_name(
            task_name=task_name, 
            algorithm_name=SoftActorCritic.ALGORITHM_NAME,
            save_dir=save_dir
            )
                
        # build the folders till save_dir as needed
        dirs_to_file = os.path.normpath(save_dir).split(os.path.sep)
        for i in range(len(dirs_to_file)):
            path_up_to_i = os.path.join(*dirs_to_file[:i+1])
            if not os.path.exists(path_up_to_i): os.mkdir(path_up_to_i)
        suffix =  ".pth"
        try: 
            torch.save(self.policy.state_dict(),    save_dir + "/policy"    + suffix)
            torch.save(self.qnet1.state_dict(),     save_dir + "/qnet1"     + suffix)
            torch.save(self.qnet2.state_dict(),     save_dir + "/qnet2"     + suffix)
            torch.save(self.qnet1_tar.state_dict(), save_dir + "/qnet1_tar" + suffix)
            torch.save(self.qnet2_tar.state_dict(), save_dir + "/qnet2_tar" + suffix)
        except:
            raise Exception("SAVING SOMEHOW FAILED...")
    
    def _get_parameter_dict(self):
        """
        Returns a dictionary of the relevant parameters to be saved for 
        this algorithm, to track saving progress.

        :return Dict algorithm_param: The parameters of the algorithm that are saved.
        """
        algorithm_param = super()._get_parameter_dict()
        algorithm_param["q_net_l_r"] = self.q_net_l_r
        algorithm_param["pol_l_r"] = self.pol_l_r
        algorithm_param["d_r"] = self.d_r
        algorithm_param["alpha"] = self.alpha
        algorithm_param["tau"] = self.tau
        algorithm_param["pol_eval_batch_size"] = self.pol_eval_batch_size
        algorithm_param["pol_imp_batch_size"] = self.pol_imp_batch_size
        algorithm_param["update_qnet_every_N_gradient_steps"] = self.update_qnet_every_N_gradient_steps
        algorithm_param["qnet_update_counter"] = self.qnet_update_counter
        algorithm_param["qnet1_loss_history"] = self.qnet1_loss_history
        algorithm_param["qnet2_loss_history"] = self.qnet2_loss_history
        algorithm_param["policy_loss_history"] = self.policy_loss_history
        algorithm_param["qnet_layer_sizes"]  = self.qnet_layer_sizes
        algorithm_param["policy_layer_sizes"]  = self.policy_layer_sizes
        return algorithm_param
    
    def load(self, path : str = None):
        """
        AS OF RIGHT NOW, THIS REQUIRES THAT THE LOADED ALGORITHM HAS IDENTICAL POLICY AND QNET 
        STRUCTURE AS THE CURRENT SELF ALGORITHM.

        ALSO, IT DOES NOT LOAD IMPORTANT INFORMATION SUCH AS OPTIMIZER, LEARNING RATES & THE ALIKE.

        Loads the group of policy & qnets to this SAC algorithm.
        loaded_sac_name is everything that specifies an algorithm up to the time of creation;
        e.g. "trained_algorithms/SAC/walker_2023_07_25_09_52"

        :param str path: The path to the directory holding the policy to be loaded, 
        defaults to None.
        An example of format will be "trained_algorithms/SAC/walker_2023_07_25_09_52/".
        """
        if not path.endswith("/"): path += "/"
        
        suffix =  ".pth"
        try: 
            self.policy.load_state_dict(   torch.load(path + "policy"    + suffix))
            self.qnet1.load_state_dict(    torch.load(path + "qnet1"     + suffix))
            self.qnet2.load_state_dict(    torch.load(path + "qnet2"     + suffix))
            self.qnet1_tar.load_state_dict(torch.load(path + "qnet1_tar" + suffix))
            self.qnet2_tar.load_state_dict(torch.load(path + "qnet2_tar" + suffix))
        except:
            raise Exception("LOADING SOMEHOW FAILED...")
        
    def _load_parameter_dict(self, dict : Dict):
        super()._load_parameter_dict(dict)
        self.__init__(q_net_learning_rate=dict["q_net_l_r"], 
                      policy_learning_rate=dict["pol_l_r"],
                      discount=dict["d_r"],
                      temperature=dict["alpha"],
                      qnet_update_smoothing_coefficient=dict["tau"],
                      pol_eval_batch_size=dict["pol_eval_batch_size"],
                      pol_imp_batch_size=dict["pol_imp_batch_size"],
                      update_qnet_every_N_gradient_steps=dict["update_qnet_every_N_gradient_steps"],
                      obs_dim_size=self.obs_dim_size,
                      act_dim_size=self.act_dim_size,
                      act_ranges=self.act_ranges, 
                      qnet_layer_sizes=dict["qnet_layer_sizes"],
                      policy_layer_sizes=dict["policy_layer_sizes"]
                      )
        self.qnet_update_counter = dict["qnet_update_counter"]
        self.qnet1_loss_history  = dict["qnet1_loss_history"]
        self.qnet2_loss_history  = dict["qnet2_loss_history"]
        self.policy_loss_history = dict["policy_loss_history"]

    def _delete_saved_algorithms(self, dir : str, task_name : str, training_id : int):
        save_dir = PolicyLearningAlgorithm.get_saving_directory_name(
            task_name=f"{task_name}_{training_id}",
            algorithm_name=SoftActorCritic.ALGORITHM_NAME,
            save_dir=f"{dir}/{task_name}_{training_id}"
            )
                
        if os.path.exists(save_dir):
            for file in os.listdir(save_dir): os.remove(os.path.join(save_dir, file))
            os.rmdir(save_dir)
    
    def show_loss_history(self, task_name : str, save_figure : bool=True, save_dir : str=None):
        """
        Plots figure indicating losses for each Q-networks and the policy
        as well as a total loss. Saves the resulting figures under save_dir, 
        if save_figure is True.

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

# OTHER USEFUL FUNCTIONS

# EXPLORATION FUNCTIONS
def no_exploration_wrapper(learning_algorithm : SoftActorCritic):
    """
    Returns an exploration function which samples actions from the 
    current policy in a stochastic way.
    To be passed as argument to trainers.
    :param SoftActorCritic learning_algorithm: The learning algorithm used to \
    sample actions.
    :return uniform_random_sampling: The function that returns the actions.
    """

    def no_exploration(obs : Union[torch.tensor, np.ndarray]):
        """
        Takes a raw observation from the environment, returning the 
        corresponding actions obtained from SAC where inference is stochastic.

        :param Union[torch.tensor, np.ndarray] obs: An observation object, either \
        of shape [obs_dim] or [batch_dim, obs_dim]. 
        :raises Exception: If obs isn't either of torch.tensor or np.ndarray, raise \
        an exception.
        :return np.ndarray: Returns the obtained action, with or without the batch \
        dimension depending on the input. 
        """
        if type(obs) == type(np.array([0])):
            obs = torch.from_numpy(obs)
        else:
            raise Exception("Value passed as action to no_exploration was of type ", type(obs), 
                            "but should be either a torch.tensor or np.ndarray to successfully work.")
        # if there is the batch dimension
        if len(obs.shape) == 2:
            return learning_algorithm(obs.to(learning_algorithm.device), deterministic=False)
        # if there is no batch dimension
        elif len(obs.shape) == 1:
            action = learning_algorithm(torch.unsqueeze(obs.to(learning_algorithm.device), dim=0), 
                                        deterministic=False)
            return np.squeeze(action, axis=0)
        else: 
            raise Exception("Given observation not of shape [batch_dim, obs_dim] or [obs_dim]...")
        
    return no_exploration


def uniform_random_sampling_wrapper(learning_algorithm : SoftActorCritic):
    """
    Returns an exploration function which uniform randomly samples
    actions from the possible environments given an SAC learning algorithm. 
    To be passed as argument to trainers.
    :param SoftActorCritic learning_algorithm: The learning algorithm, passed \
    to get data on act_dim_size, action_multipler and action_avgs.
    :return uniform_random_sampling: The function that returns the actions.
    """
    def uniform_random_sampling(obs : Union[torch.tensor, np.ndarray]):
        """
        Returns an action uniform randomly sampled from possible actions in the world.
        :param Union[torch.tensor, np.ndarray] obs: An observation object, either \
        of shape [obs_dim] or [batch_dim, obs_dim].
        :return np.ndarray: Returns the obtained action, with or without the batch \
        dimension depending on the input. 
        """
        # if there is the batch dimension
        if len(obs.shape) == 2:
            action_zero_to_one = torch.rand(
                size=(obs.shape[0], #batch size
                      learning_algorithm.act_dim_size)
                ).cpu()
        # if there is no batch dimension
        elif len(obs.shape) == 1:
            action_zero_to_one = torch.rand(size=(learning_algorithm.act_dim_size,)).cpu()
        else: 
            raise Exception("Given observation not of shape [batch_dim, obs_dim] or [obs_dim]...")
        
        action_minus_one_to_one = action_zero_to_one * 2.0 - 1.0
        adjusted_actions = (action_minus_one_to_one * 
                            learning_algorithm.policy.action_multiplier.detach().cpu() + 
                            learning_algorithm.policy.action_avgs.detach().cpu())
        return adjusted_actions.numpy()
    
    return uniform_random_sampling