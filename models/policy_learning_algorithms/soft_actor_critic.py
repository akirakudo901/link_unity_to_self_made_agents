"""
A trial implementation of Soft-Actor Critic as documented in the spinning-up page:
https://spinningup.openai.com/en/latest/algorithms/sac.html
"""

from datetime import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from models.policy_learning_algorithms.policy_learning_algorithm import OffPolicyLearningAlgorithm
from models.trainers.gym_base_trainer import Buffer

class SoftActorCritic(OffPolicyLearningAlgorithm):

    NUMBER_DTYPE = torch.float32
    
    class QNet(nn.Module):
        """
        A neural network approximating the q-value function.
        """
        
        def __init__(self, observation_size : int, action_size : int):
            """
            Initializes one of the Q-function approximating NN which evaluates
            the value of any state-action pair.
            
            In the original SAC paper found at this page[https://arxiv.org/pdf/1801.01290.pdf],
            the following are set as found in D. hyperparameters:
            - number of hidden layers = 2
            - number of hidden units per layer = 256
            - nonlinearity = ReLU

            :param int observation_size: The size of the observation input vector.
            :param int action_size: The size of the action vectors.
            """
            super(SoftActorCritic.QNet, self).__init__()
            fc_out = 256
            self.fc1 = nn.Linear(observation_size + action_size, fc_out)
            self.fc2 = nn.Linear(fc_out, fc_out)

            self.linear_relu_stack = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU()
            )

            self.last_fc = nn.Linear(fc_out, 1)
        
        def forward(self, obs : torch.tensor, actions : torch.tensor):
            """
            Passes obs and actions vector (or batches of them) into the nn to 
            return a value indicating each state-action pairs' q-value.

            :param torch.tensor obs: Batch of state vectors in the form [batch, obs_dim].
            :param torch.tensor actions: Batch of action vectors in the form [batch, act_dim].
            :return torch.tensor out: Q-values for each batch of state-action pair.
            """
            # concatenate input skipping batch dimension to input to nn
            inp = torch.cat((obs, actions), dim=1)
            # passing it through the NN
            x = self.linear_relu_stack(inp)
            out = self.last_fc(x)
            return out

    class Policy(nn.Module):
        """
        A neural network approximating the policy.
        Composed of a stochastic noise component and a deterministic component
        which maps the noises to corresponding actions - which is required to apply the 
        reparameterization trick, as proposed in the original paper.
        """
        """
        We will start by modeling the noise & deterministic components according
        to the original SAC paper and some imagination of my own:
        - noise as sampled from a spherical gaussian distribution
        - deterministic component as a combination of a transformation based on mean and sd 
         given by evaluating the neural network at observation - the multivariate normal 
         distribution is assumed to have a diagonal covariance matrix for simplicity (and as 
         it still seems to work well enough from reading the paper) - and tanh to limit possible
         action values.
        """

        def __init__(self, observation_size : int, action_size : int, action_ranges : Tuple[Tuple[float]]):
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
            """
            super(SoftActorCritic.Policy, self).__init__()
            self.action_avgs       = torch.tensor([(range[1] + range[0]) / 2 for range in action_ranges]).to(SoftActorCritic.NUMBER_DTYPE)
            self.action_multiplier = torch.tensor([(range[1] - range[0]) / 2 for range in action_ranges]).to(SoftActorCritic.NUMBER_DTYPE)

            fc_out = 256
            self.fc1 = nn.Linear(observation_size, fc_out)
            self.fc2 = nn.Linear(fc_out, fc_out)
            
            self.linear_relu_stack = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU()
            )

            #TODO experimentation around whether constraining mean between -1 to 1 * action multiplier
            # allows training to keep going without gradient falling to inf
            self.mean_layer = nn.Linear(fc_out, action_size) #initial implementation
            # self.mean_layer = nn.Sequential(
            #     nn.Linear(fc_out, action_size),
            #     nn.Tanh()
            #     )
            self.sd_layer = nn.Linear(fc_out, action_size)

            # TODO TO REMOVE
            self.std_history, self.mean_history, self.stackout_history = [], [], []
            self.history_obs = None
            # END TO REMOVE

        def forward(self, 
                    obs : torch.tensor,
                    num_samples : int = 1,
                    deterministic : bool = False
                    ):
            """
            Takes in the observation to spit out num_samples actions sampled from the 
            policy, as well as their log probabilities.
            The policy is approximated by a gaussian with mean/sd obtained by evaluating
            a neural network at observation.
            
            *num_samples seems to be set to 1 in the original implementation from the first 
            SAC paper & in the Spinning Up implementation; I might not have a concrete
            understanding yet of why that is the choice (instead of, let's say, sampling
            100 actions and calculate them all for the given task).
            *I am also a bit confused as to why we simply compute the log probability of 
            the action at the time, rather than directly calculating the entropy of the
            gaussian, which we have at hand.

            :param torch.tensor obs: The observations at which the policy is evaluated.
            :param int num_samples: The number of samples we obtain for each observation.
            :param bool deterministic: Whether the return should be deterministic (only at inference)

            if deterministic is True:
            :return torch.tensor myus: Actions obtained as means of the gaussian distributions which
            are used as estimates of the policy at the corresponding states. The result is squashed 
            and adjusted to match the action_ranges values.

            if deterministic is False:
            :return torch.tensor squashed: The actions sampled from the policy and squashed between
            -1 and 1 using the tanh (as done with the original paper to limit possible action range)
            before being adjusted to match the action_ranges values.
            :return torch.tensor log_probs: The log probability for corresponding actions in squashed
            after appropriate adjustment.
            """
            
            # we squash action values to be between a given bounary using tanh and adjustment
            def squashing_function(actions : torch.tensor):
                # print("actions right before tanh: ", actions, "actions.shape: ", actions.shape)
                squashed_neg_one_to_one = torch.tanh(actions)
                # print("actions right after tanh: ", squashed_neg_one_to_one, "squashed_neg_one_to_one.shape: ", squashed_neg_one_to_one.shape)
                t_device = squashed_neg_one_to_one.device
                # print("action_multiplier: ", self.action_multiplier, "action_avgs: ", self.action_avgs)
                return (squashed_neg_one_to_one * 
                        self.action_multiplier.to(t_device).detach() + 
                        self.action_avgs.to(t_device).detach())

            # TODO TO REMOVE
            if self.history_obs == None:
                self.history_obs = obs[:3]
            # END TO REMOVE

            # we obtain the mean myu and sd sigma of the gaussian distribution
            stack_out = self.linear_relu_stack(obs)

            hist_stackout = self.linear_relu_stack(self.history_obs) # TODO TO REMOVE
            self.stackout_history.append(
                [hist_stackout[i].detach().numpy() for i in range(self.history_obs.shape[0])]
                ) # TODO TO REMOVE
            
            myus = self.mean_layer(stack_out)

            hist_mean = self.mean_layer(hist_stackout) # TODO TO REMOVE
            self.mean_history.append(
                [hist_mean[i].detach().numpy() for i in range(self.history_obs.shape[0])]
                ) # TODO TO REMOVE

            #TODO EXPERIMENTAL
            # sigmas = torch.abs(self.sd_layer(stack_out)) #TODO can we use abs here? Or is exp better?
            # sigmas = torch.exp(self.sd_layer(stack_out)) #squash to enforce positive sigmas values

            # ONE WAY TO DO IT (FROM SPINNING UP?)
            LOG_STD_MIN, LOG_STD_MAX = -5, 2
            log_std = torch.tanh(self.sd_layer(stack_out)) # another way to do it: tanh and scaling
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            sigmas = torch.exp(log_std)

            hist_sigma = self.sd_layer(hist_stackout) # TODO TO REMOVE
            self.std_history.append(
                [hist_sigma[i].detach().numpy() for i in range(self.history_obs.shape[0])]
                ) # TODO TO REMOVE
            # ONE WAY TO DO IT END

            # ANOTHER WAY TO DO IT (FROM https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py)
            # LOG_STD_MAX = 2
            # LOG_STD_MIN = -20

            # sigmas = torch.exp(torch.clamp(self.sd_layer(stack_out), min=LOG_STD_MIN, max=LOG_STD_MAX))
            #END EXPERIMENTAL
            
            # print("myus: ", myus, "myus.shape: ", myus.shape)
            # print("sigmas: ", sigmas, "sigmas.shape: ", sigmas.shape)
            
            # if deterministic (while in inference), return the mean of distributions
            # corresponding to actions at time of inference, but squashed as needed
            if deterministic: return squashing_function(myus)
            
            # then evaluate the probability that action is chosen under the distribution
            dist = Normal(loc=myus, scale=sigmas) #MultivariateNormal with diagonal covariance
            actions_num_samples_first = dist.rsample(sample_shape=(num_samples, )).to(SoftActorCritic.NUMBER_DTYPE)
            actions = torch.transpose(actions_num_samples_first, dim0=0, dim1=1)
            # print("actions: ", actions, "actions.shape: ", actions.shape)
            squashed = squashing_function(actions)
            # print("squashed: ", squashed, "squashed.shape: ", squashed.shape)

            # pure_probabilities is log_prob for each action when sampled from each normal distribution
            # aggregating over the entire action, it becomes their sum (product of independent events but logged)
            pure_log_probabilities = torch.transpose(dist.log_prob(actions_num_samples_first).to(SoftActorCritic.NUMBER_DTYPE), dim0=0, dim1=1)
            # print("pure_log_probabilities: ", pure_log_probabilities, "pure_log_probabilities.shape: ", pure_log_probabilities.shape)
            before_correction = torch.sum(pure_log_probabilities, dim=2)
            # print("before_correction: ", before_correction, "before_correction.shape: ", before_correction.shape)
            log_probs = self._correct_for_squash(
                before_correction, actions
                )
            
            # TODO remove
            # if (squashed.isinf().any() or squashed.isnan().any() or 
            #     log_probs.isinf().any() or log_probs.isnan().any()):
            #     print("Inside policy's forward!")
            #     print(f"obs is {obs}\n.")
            #     print(f"myus is {myus}\n.")
            #     print(f"sigmas is {sigmas}\n.")
                # print(f"actions is {actions}\n.")
                # print(f"squashed is {squashed}\n.")
                # print(f"pure_log_probabilities is {pure_log_probabilities}\n.")
                # print(f"before_correction is {before_correction}\n.")
                # print(f"log_probs is {log_probs}\n.")
                # print("Done with policy's forward!")
            
            return squashed, log_probs
        
        # TODO REMOVE
        def show_histories_for_debugs(self):
            def show_plot_and_save(history, nn):
                plt.clf()
                plt.plot(range(0, len(history)), history)
                plt.savefig(f"{nn}_loss_history_fig.png")
                plt.show()
            
            all_std_history = list(zip(*self.std_history))
            all_stackout_history = list(zip(*self.stackout_history))
            all_mean_history = list(zip(*self.mean_history))

            for h, nn in (
                [(all_std_history[i],  f"std_{i}") for i in range(len(all_std_history))] +  
                [(all_stackout_history[i],  f"stackOut_{i}") for i in range(len(all_stackout_history))] + 
                [(all_mean_history[i],  f"mean_{i}") for i in range(len(all_mean_history))]
            ):
                show_plot_and_save(h, nn)
        
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

            Furthermore, as we scale the action by different factors after squashing with tanh to adjust
            to the actual action ranges, we multiply the whole result by self.action_multiplier before the sum.

            :param torch.tensor before: The log probabilities to be corrected.
            Will be of the form [batch_size, num_samples].
            :param torch.tensor actions: The actions which log probabilities are in question.
            Will be of the form [batch_size, num_samples, action_size].
            :return torch.tensor after: The log probabilities after being corrected.
            Will be of the form [batch_size, num_samples].
            """
            # compute the trace of the Jacobian of tanh squashing for each action
            # print("before: ", before, "before.shape: ", before.shape)
            # print("actions[:10]: ", actions[:10], "actions.shape: ", actions.shape)
            multiplier = self.action_multiplier.to(actions.device)
            # TODO EXPERIMENTAL
            # jacobian_trace = torch.sum(torch.log(multiplier * (1 - torch.tanh(actions).pow(2)) + 1e-6), dim=2)
            
            # apparently below is numerically equivalent to above but more stable:
            # https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py#L73 
            jacobian_trace = torch.squeeze((torch.log(torch.tensor(2.)) + 
                                            2. * 
                                            (torch.log(multiplier) - 
                                             actions - 
                                             torch.nn.functional.softplus(-2. * actions))), dim=1) 
            # EXPERIMENTAL END
            # print("jacobian_trace.shape: ", jacobian_trace.shape)
            
            # jacobian_trace = torch.sum((multiplier * torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)), dim=2) #old code with mistake in position of multipler
            # print("jacobian_trace: ", jacobian_trace, "jacobian_trace.shape: ", jacobian_trace.shape)
            # subtract it from before to yield after
            after = before - jacobian_trace
            # print("after: ", after, "after.shape: ", after.shape)
            
            return after

    def __init__(self,
                 q_net_learning_rate : float,
                 policy_learning_rate : float, 
                 discount : float,
                 temperature : float,
                 qnet_update_smoothing_coefficient : float,
                 observation_size : int,
                 action_size : int,
                 action_ranges : Tuple[Tuple[float]],
                 pol_eval_batch_size : int,
                 pol_imp_batch_size : int,
                 update_qnet_every_N_gradient_steps : int,
                 optimizer : optim.Optimizer = optim.Adam
                 ):
        self.q_net_l_r = q_net_learning_rate
        self.pol_l_r = policy_learning_rate
        self.d_r = discount
        self.alpha = temperature
        self.tau = qnet_update_smoothing_coefficient
        self.obs_size = observation_size
        self.act_size = action_size
        self.act_ranges = action_ranges
        self.pol_eval_batch_size = pol_eval_batch_size
        self.pol_imp_batch_size = pol_imp_batch_size
        self.update_qnet_every_N_gradient_steps = update_qnet_every_N_gradient_steps
        
        self.device = self.set_device()

        self.qnet_update_counter = 1
        self.TEMP_policy_counter = 1 #TODO

        # two q-functions are used to approximate values during training
        self.qnet1 = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.qnet2 = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.optim_qnet1 = optimizer(self.qnet1.parameters(), lr=q_net_learning_rate)
        self.optim_qnet2 = optimizer(self.qnet2.parameters(), lr=q_net_learning_rate)
        # two target networks are updated less (e.g. every 1000 steps) to compute targets for q-function update
        self.qnet1_tar = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.qnet2_tar = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        # transfer the weights to sync
        self._update_target_networks(hard_update=True)
        
        # initialize policy 
        self.policy = SoftActorCritic.Policy(
            observation_size=observation_size, action_size=action_size, action_ranges=action_ranges
            ).to(self.device)
        self.optim_policy = optimizer(self.policy.parameters(), lr=policy_learning_rate)

        self.qnet1_loss_history = [] #List of float
        self.qnet2_loss_history = [] #List of float
        self.policy_loss_history = [] #List of float
    
    def set_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        return device
    
    def __call__(self, state):
        """
        A format put in place where SAC is called raw to obtain actions from it.

        :param state: The state of observation in question to which this algorithm was applied.
        Could be a numpy array or torch tensor? TODO ASCERTAIN!
        """
        return self.get_optimal_action(state)
    
    def update(self, experiences : Buffer, seed=None):
        # a single experience contained in "experiences" is of the form: 
        # (obs, action, reward, done, next_obs)

        # TODO THIS IS EXPERIMENTAL - REMOVE AS NEEDED
        # def clip_gradient(net: nn.Module) -> None:
        #     for param in net.parameters():
        #         param.grad.data.clamp_(-1, 1)
        #EXPERIMENTAL END

        NUM_EVAL_STEPS = 1

        POLICY_EVAL_NUM_EPOCHS = 1
        POL_EVAL_FRESH_ACTION_SAMPLE_SIZE = 1

        NUM_IMP_STEPS = 1

        POLICY_IMP_NUM_EPOCHS = 1
        POL_IMP_FRESH_ACTION_SAMPLE_SIZE = 1

        observations, actions, rewards, dones, next_observations = SoftActorCritic._sample_experiences(
            experiences=experiences, num_samples=POLICY_EVAL_NUM_EPOCHS * self.pol_eval_batch_size, 
            device=self.device, seed=seed
            )
        # print("observations.shape: ", observations.shape)
        # print("actions.shape: ", actions.shape)
        # print("rewards.shape: ", rewards.shape)
        # print("dones.shape: ", dones.shape)
        # print("next_observations.shape: ", next_observations.shape)
        
        # freshly sample new actions in the current policy for each next observations
        # for now, we will sample FRESH_ACTION_SAMPLE_SIZE
        # TODO WHAT IS THE BEST WAY TO FRESHLY SAMPLE THOSE?
        fresh_action_samples, fresh_log_probs = self.policy(next_observations, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        # if fresh_action_samples.isinf().any() or fresh_action_samples.isnan().any():
        #     print("After fresh_action_samples!")
        #     print("next_observations: ", next_observations, "\n")
            # print("next_observations.shape: ", next_observations.shape, "\n")
            # print("fresh_action_samples: ", fresh_action_samples, "\n")
            # print("fresh_action_samples.shape: ", fresh_action_samples.shape, "\n")
            # print("fresh_log_probs: ", fresh_log_probs, "\n")
            # print("fresh_log_probs.shape: ", fresh_log_probs.shape, "\n")
            # print("Done with fresh_action_samples!")
        
        # print("fresh_action_samples: ", fresh_action_samples, "\n")
        # print("fresh_action_samples.shape: ", fresh_action_samples.shape, "\n")
        # print("fresh_log_probs: ", fresh_log_probs, "\n")
        # print("fresh_log_probs.shape: ", fresh_log_probs.shape, "\n")

        # 1 - policy evaluation
        policy_evaluation_gradient_step_count = 0

        for _ in range(POLICY_EVAL_NUM_EPOCHS):
            for i in range(experiences.size() // self.pol_eval_batch_size + 1):
                batch_start = i*self.pol_eval_batch_size
                batch_end = min((i+1)*self.pol_eval_batch_size, experiences.size())

                batch_obs = observations[batch_start : batch_end].detach()
                # print("batch_obs[:10]: ", batch_obs[:10], "\n")
                # print("batch_obs.shape: ", batch_obs.shape, "\n")
                batch_actions = actions[batch_start : batch_end].detach()
                # print("batch_actions[:10]: ", batch_actions[:10], "\n")
                # print("batch_actions.shape: ", batch_actions.shape, "\n")
                batch_rewards = rewards[batch_start : batch_end].detach()
                # print("batch_rewards[:10]: ", batch_rewards[:10], "\n")
                # print("batch_rewards.shape: ", batch_rewards.shape, "\n")
                batch_dones = dones[batch_start : batch_end].detach()
                # print("batch_dones[:10]: ", batch_dones[:10], "\n")
                # print("batch_dones.shape: ", batch_dones.shape, "\n")
                batch_nextobs = next_observations[batch_start : batch_end].detach()
                # print("batch_nextobs[:10]: ", batchbatch_nextobs_dones[:10], "\n")
                # print("batch_nextobs.shape: ", batch_nextobs.shape, "\n")

                # batch_action_samples' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, action_size]
                batch_action_samples = fresh_action_samples[batch_start : batch_end]
                # print("batch_action_samples[:10]: ", batch_action_samples[:10], "\n")
                # print("batch_action_samples.shape: ", batch_action_samples.shape, "\n")
                
                # batch_log_probs' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE]
                batch_log_probs = fresh_log_probs[batch_start : batch_end]
                # print("batch_log_probs[:10]: ", batch_log_probs[:10], "\n")
                # print("batch_log_probs.shape: ", batch_log_probs.shape, "\n")
                
                # first compute target value for all experiences (terminal ones are only the rewards)
                targets = self._compute_qnet_target(batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs)
                
                # then compute prediction value for all experiences
                predictions1 = torch.squeeze(self.qnet1(obs=batch_obs, actions=batch_actions), dim=1)
                predictions2 = torch.squeeze(self.qnet2(obs=batch_obs, actions=batch_actions), dim=1)
                
                # finally take loss through MSELoss
                criterion = nn.MSELoss()
                loss1 = criterion(predictions1, targets)
                loss2 = criterion(predictions2, targets)

                self.qnet1_loss_history.append(loss1.item())
                self.qnet2_loss_history.append(loss2.item())
                # backpropagate that loss to update q_nets
                self.optim_qnet1.zero_grad()
                loss1.backward()
                # clip_gradient(self.qnet1) #TODO EXPERIMENTAL REMOVE AS NEEDED
                self.optim_qnet1.step()

                self.optim_qnet2.zero_grad()
                loss2.backward()
                # clip_gradient(self.qnet2) #TODO EXPERIMENTAL REMOVE AS NEEDED
                self.optim_qnet2.step()
                
                # increment the update counter and update target networks every N gradient steps
                self.qnet_update_counter += 1
                if self.qnet_update_counter % self.update_qnet_every_N_gradient_steps == 0:
                    self._update_target_networks(hard_update=False)

                    
                    # print("Updated qnet!\n") #TODO REMOVE
                    # print("qnet targets[:10]: ", targets[:10], "\nqnet targets.shape: ", targets.shape, "\n")
                    # print("qnet predictions1[:10]: ", predictions1[:10], "\nqnet predictions1.shape: ", predictions1.shape, "\n")
                    # print("qnet loss1: ", loss1, "\nqnet loss1.shape: ", loss1.shape, "\n")
                def went_to_nan_or_inf(tensor, name):
                    if not (tensor.isnan().any() or tensor.isinf().any()):
                        return
                    
                    print()
                    if tensor.isnan().any():
                        raise Exception(f"{name} went to NaN!")
                    elif tensor.isinf().any():
                        raise Exception(f"{name} went to Inf!")
                    
                went_to_nan_or_inf(targets, "targets")
                went_to_nan_or_inf(predictions1, "predictions1")
                went_to_nan_or_inf(loss1, "loss1")
                went_to_nan_or_inf(loss2, "loss2")
                    
                
                # finally break out of loop if the number of gradient steps exceeds NUM_STEPS
                policy_evaluation_gradient_step_count += 1
                if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break
            # also break out of outer loop
            if policy_evaluation_gradient_step_count >= NUM_EVAL_STEPS: break

        # 2 - policy improvement
        # at this point, the q-network weights are adjusted to reflect the q-value of
        # the current policy. We just have to take a gradient step with respect to the  
        # distance between this q-value and the current policy

        new_seed = ((seed - 1)*seed) % (seed + 1) if seed is not None else None
        observations, actions, rewards, dones, next_observations = SoftActorCritic._sample_experiences(
            experiences=experiences, num_samples=POLICY_IMP_NUM_EPOCHS * self.pol_imp_batch_size, 
            device=self.device, seed=new_seed
            )
        
        # in order to estimate the gradient, we again sample some actions at this point in time.
        # TODO COULD WE USE ACTIONS SAMPLED BEFORE WHICH WERE USED FOR Q-NETWORK UPDATE? NOT SURE
        fresh_action_samples2, fresh_log_probs2 = self.policy(observations, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        
        policy_improvement_gradient_step_count = 0

        for _ in range(POLICY_IMP_NUM_EPOCHS):
            for i in range(experiences.size() // self.pol_imp_batch_size + 1):
                batch_start = i*self.pol_imp_batch_size
                batch_end = min((i+1)*self.pol_imp_batch_size, experiences.size())

                batch_obs = observations[batch_start : batch_end].detach()
                # print(batch_obs, "\n")
                
                # batch_action_samples' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, action_size]
                batch_action_samples = fresh_action_samples2[batch_start : batch_end]
                # print("batch_action_samples: ", batch_action_samples, "\n")
                # print("batch_action_samples.size: ", batch_action_samples.size, "\n")
                
                # batch_log_probs' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE]
                batch_log_probs = fresh_log_probs2[batch_start : batch_end]
                # print("batch_log_probs: ", batch_log_probs, "\n")
                # print("batch_log_probs.size: ", batch_log_probs.size, "\n")
                
                # Then, we compute the loss function: which relates the exponentiated distribution of
                # the q-value function with the current policy's distribution, through KL divergence
                # target_exped_qval = self._compute_exponentiated_qval(batch_obs, batch_action_samples) #TODO EXPEIRMENTAL
                target_qval = self._compute_policy_val(batch_obs, batch_action_samples) #TODO EXPEIRMENTAL
                policy_val = torch.mean(batch_log_probs, dim=1)

                #TODO EXPERIMENTAL TRYING OUT THE CLEAN RL LOSS FUNCTION
                # BELOW IS MY ORIGINAL APPROACH
                # criterion = nn.KLDivLoss(reduction="batchmean")
                # loss = criterion(policy_val, target_exped_qval)

                # BELOW IS CLEAN RL
                loss = ((self.alpha * policy_val) - target_qval).mean()

                #END EXPERIMENTAL
                
                self.policy_loss_history.append(loss.item())
                
                # then, we improve the policy by minimizing this loss 
                self.optim_policy.zero_grad()
                loss.backward()
                # clip_gradient(self.policy) #TODO EXPERIMENTAL REMOVE AS NEEDED

                if torch.stack([torch.isnan(p.grad.norm()).any() for p in self.policy.parameters()]).any():
                    print(f"batch_obs: {batch_obs}, shape: {batch_obs.shape}")
                    print(f"batch_action_samples: {batch_action_samples}, shape: {batch_action_samples.shape}")
                    print(f"batch_log_probs: {batch_log_probs}, shape: {batch_log_probs.shape}")
                    print(f"policy_val: {policy_val}, shape: {policy_val.shape}")
                    print(f"target_qval: {target_qval}, shape: {target_qval.shape}")
                    print(f"loss: {loss}, shape: {loss.shape}")
                    
                    for p in self.policy.parameters():#TODO REMOVE
                        print(p.grad)#TODO REMOVE
                    raise Exception("Some gradients were NaN!")

                self.optim_policy.step()

                if torch.stack([torch.isnan(p).any() for p in self.policy.parameters()]).any(): #TODO REMOVE
                    print(list(self.policy.parameters()))#TODO REMOVE
                    raise Exception("Some of the model parameters went NaN!")#TODO REMOVE

                # TODO REMOVE
                self.TEMP_policy_counter += 1 #TODO
                # if self.TEMP_policy_counter % self.update_qnet_every_N_gradient_steps == 0: 
                    # print("See policy!", "\n")
                    # # print("policy target_exped_qval[:10]: ", target_exped_qval[:10], "\npolicy target_exped_qval.shape: ", target_exped_qval.shape, "\n")
                    # print("policy target_qval[:10]: ", target_qval[:10], "\npolicy target_qval.shape: ", target_qval.shape, "\n")
                    # print("policy_val[:10]: ", policy_val[:10], "\npolicy_val.shape: ", policy_val.shape, "\n")
                    # print("policy loss: ", loss, "\npolicy loss.shape: ", loss.shape, "\n")                        
                # went_to_nan_or_inf(target_qval, "target_qval")
                # went_to_nan_or_inf(policy_val, "policy_val")
                # went_to_nan_or_inf(loss, "loss")                
                    # END TOREMOVE

                # finally increment the improvement gradient step count
                policy_improvement_gradient_step_count += 1
                if policy_improvement_gradient_step_count >= NUM_IMP_STEPS: break
            # also break out of outer loop
            if policy_improvement_gradient_step_count >= NUM_IMP_STEPS: break
                
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
        # print("qnet1_val: ", qnet1_val, "\n", "qnet1_val.shape: ", qnet1_val.shape, "\n")
        # the minimum of those predictions of the shape [batch, num_samples]
        qval_minimum = torch.minimum(qnet1_val, qnet2_val)
        # print("qval_minimum: ", qval_minimum, "\n", "qval_minimum.shape: ", qval_minimum.shape, "\n")
        # exp_qval_minimum = torch.exp(qval_minimum)
        # print("exp_qval_minimum: ", exp_qval_minimum, "\n", "exp_qval_minimum.shape: ", exp_qval_minimum.shape, "\n")
        # the mean of those predictions of the shape [batch]
        mean_exp_qval = torch.mean(qval_minimum, dim=1, dtype=SoftActorCritic.NUMBER_DTYPE)
        # print("mean_exp_qval: ", mean_exp_qval, "\n", "mean_exp_qval.shape: ", mean_exp_qval.shape, "\n")

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
            # print("qnet1_tar_preds: ", qnet1_tar_preds, "\n", "qnet1_tar_preds.shape: ", qnet1_tar_preds.shape, "\n")
            qnet2_tar_preds = torch.cat([
                self.qnet2_tar(batch_nextobs, torch.squeeze(batch_action_samples[:, i, :], dim=0))
                for i in range(batch_action_samples.shape[1])
                ], dim=1)
            # print("qnet2_tar_preds: ", qnet2_tar_preds, "\n")
            # print("qnet2_tar_preds.shape: ", qnet2_tar_preds.shape, "\n")
            minimum = torch.minimum(qnet1_tar_preds, qnet2_tar_preds)
            # print("minimum: ", minimum, "\n")
            # print("minimum.shape: ", minimum.shape, "\n")
            mean_of_minimum = torch.mean(minimum, dim=1, dtype=SoftActorCritic.NUMBER_DTYPE)
            # print("mean_of_minimum: ", mean_of_minimum, "\n")
            # print("mean_of_minimum.shape: ", mean_of_minimum.shape, "\n")
            mean_of_log = torch.mean(batch_log_probs, dim=1)
            # print("mean_of_log: ", mean_of_log, "\n")
            # print("mean_of_log.shape: ", mean_of_log.shape, "\n")
            targets = (
                        batch_rewards + 
                        self.d_r *
                        (1.0 - batch_dones.to(SoftActorCritic.NUMBER_DTYPE)) *
                        (mean_of_minimum - self.alpha * mean_of_log)
                    )
            # print("targets: ", targets, "\n")
            # print("targets.shape: ", targets.shape, "\n")

            # TODO remove
            # if targets.isinf().any() or targets.isnan().any():
            #     print("Inside _compute_target!")
                    # print(f"batch_nextobs is: {batch_nextobs}.\n")
                    # print(f"batch_action_samples is: {batch_action_samples}.\n")
                    # print(f"batch_log_probs is: {batch_log_probs}.\n")
            #     # print(f"qnet1_tar_preds is: {qnet1_tar_preds}.\n")
            #     # print(f"qnet2_tar_preds is: {qnet2_tar_preds}.\n")
            #     # print(f"minimum is: {minimum}.\n")
            #     # print(f"mean_of_minimum is: {mean_of_minimum}.\n")
            #     # print(f"mean_of_log is: {mean_of_log}.\n")
            #     # print(f"targets is: {targets}.\n")
            #     print("Done with _compute_target!")

        return targets
    
    def _update_target_networks(self, hard_update : bool = False):
        """
        Updates the target networks utilizing the target smoothing coefficient 
        to slowly adjust parameters by considering from the corresponding q-networks.
        """
        if hard_update:
            self.qnet1_tar.load_state_dict(self.qnet1.state_dict())
            self.qnet2_tar.load_state_dict(self.qnet2.state_dict())
        else:
            for param, target_param in zip(self.qnet1.parameters(), self.qnet1_tar.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qnet2.parameters(), self.qnet2_tar.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _unzip_experiences(experiences : Buffer, device = None):
        """
        Unzips the experiences into groups of observations, 
        actions, rewards, done flags, and next observations, to be returned.

        :param Buffer experiences: A Buffer containing obs, action, reward, done and next_obs.
        :return Tuple[torch.tensor]: Tensors of each component in the Buffer; 
        observations, actions, rewards, dones, next_observations.
        """
        np_obs, np_act, np_rew, np_don, np_next_obs = experiences.get_components()
        observations, actions, rewards, dones, next_observations = (torch.from_numpy(np_obs).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_act).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_rew).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_don),
                                                                    torch.from_numpy(np_next_obs).to(SoftActorCritic.NUMBER_DTYPE))
        if device != None:
            observations, actions, rewards, dones, next_observations = (torch.from_numpy(np_obs).to(device),
                                                                        torch.from_numpy(np_act).to(device),
                                                                        torch.from_numpy(np_rew).to(device),
                                                                        torch.from_numpy(np_don).to(device),
                                                                        torch.from_numpy(np_next_obs).to(device))
        return observations,actions,rewards,dones,next_observations
    
    def _sample_experiences(experiences : Buffer, num_samples : int, device = None, seed : int = None):
        """
        Randomly samples num_samples experiences from the given experiences buffer, expanding them 
        into observations, actions, rewards, done flags, and next observations, to be returned.

        :param Buffer experiences: A Buffer containing obs, action, reward, done and next_obs.
        :param int num_samples: The number of random samples we get from the buffer.
        :param device: The device to which we move resulting tensors, if any.
        :param int seed: The seed used for random sampling of experiences, if specified.
        Not used if None.
        :return Tuple[torch.tensor]: Tensors of each component in the Buffer; 
        observations, actions, rewards, dones, next_observations.
        """
        np_obs, np_act, np_rew, np_don, np_next_obs = experiences.sample_random_experiences(num_samples=num_samples, seed=seed)
        observations, actions, rewards, dones, next_observations = (torch.from_numpy(np_obs).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_act).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_rew).to(SoftActorCritic.NUMBER_DTYPE),
                                                                    torch.from_numpy(np_don),
                                                                    torch.from_numpy(np_next_obs).to(SoftActorCritic.NUMBER_DTYPE))
        if device != None:
            observations, actions, rewards, dones, next_observations = (torch.from_numpy(np_obs).to(device),
                                                                        torch.from_numpy(np_act).to(device),
                                                                        torch.from_numpy(np_rew).to(device),
                                                                        torch.from_numpy(np_don).to(device),
                                                                        torch.from_numpy(np_next_obs).to(device))
        return observations,actions,rewards,dones,next_observations
    
    def get_optimal_action(self, state):
        """
        Computes the currently optimal action given an observation state.
        State can be either a torch tensor or numpy array, both being converted
        into a torch tensor before further processing is done.

        :param state: The observation state given as torch.tensor or np.ndarray.
        :return np.ndarray action: The action the policy deems optimal as ndarray. 
        """
        # if type is not torch.tensor, try casting
        if type(state) != type(torch.tensor([0])):
            try:
                state = torch.from_numpy(state)
            except:
                raise Exception("Error in reading observation within SAC get_optimal_action; \
                                'state' needs to be one of torch.tensor or np.ndarray.")
        action = self.policy(obs=state.to(self.device), deterministic=True)
        return action.detach().numpy()
    
    def save(self, task_name: str, save_dir : str = None):
        creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if save_dir is None:
            save_dir = f"trained_algorithms/SAC/{task_name}_{creation_time}"
        
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        suffix =  ".pth"
        try: 
            torch.save(self.policy.state_dict(),    save_dir + "/policy"    + suffix)
            torch.save(self.qnet1.state_dict(),     save_dir + "/qnet1"     + suffix)
            torch.save(self.qnet2.state_dict(),     save_dir + "/qnet2"     + suffix)
            torch.save(self.qnet1_tar.state_dict(), save_dir + "/qnet1_tar" + suffix)
            torch.save(self.qnet2_tar.state_dict(), save_dir + "/qnet2_tar" + suffix)
        except:
            raise Exception("SAVING SOMEHOW FAILED...")
    
    def load(self, loaded_sac_name : str = None):
        """
        AS OF RIGHT NOW, THIS REQUIRES THAT THE LOADED ALGORITHM HAS IDENTICAL POLICY AND QNET 
        STRUCTURE AS THE CURRENT SELF ALGORITHM.

        ALSO, IT DOES NOT LOAD IMPORTANT INFORMATION SUCH AS OPTIMIZER, LEARNING RATES & THE ALIKE.

        Loads the group of policy & qnets to this SAC algorithm.
        loaded_sac_name is everything that specifies an algorithm up to the time of creation;
        e.g. "trained_algorithms/SAC/walker_2023_07_25_09_52"

        :param str loaded_sac_name: The name of the directory holding the policy to be loaded, 
        defaults to None.
        An example of format will be "trained_algorithms/SAC/walker_2023_07_25_09_52/".
        """
        if not loaded_sac_name.endswith("/"): loaded_sac_name += "/"
        
        suffix =  ".pth"
        try: 
            self.policy.load_state_dict(   torch.load(loaded_sac_name + "policy"    + suffix))
            self.qnet1.load_state_dict(    torch.load(loaded_sac_name + "qnet1"     + suffix))
            self.qnet2.load_state_dict(    torch.load(loaded_sac_name + "qnet2"     + suffix))
            self.qnet1_tar.load_state_dict(torch.load(loaded_sac_name + "qnet1_tar" + suffix))
            self.qnet2_tar.load_state_dict(torch.load(loaded_sac_name + "qnet2_tar" + suffix))
        except:
            raise Exception("LOADING SOMEHOW FAILED...")
    
    def show_loss_history(self, task_name):
        def show_plot_and_save(history, nn):
            plt.clf()
            plt.plot(range(0, len(history)), history)
            plt.savefig(f"{task_name}_{nn}_loss_history_fig.png")
            plt.show()
        
        try:
            total_loss_history = [(self.qnet1_loss_history[i] + 
                              self.qnet2_loss_history[i] + 
                              self.policy_loss_history[i])
                              for i in range(min(
                                  len(self.qnet1_loss_history),
                                  len(self.qnet2_loss_history),
                                  len(self.policy_loss_history))
                                  )]
        except:
            total_loss_history = []
        
        for h, nn in [(self.qnet1_loss_history,  "qnet1"), 
                      (self.qnet2_loss_history,  "qnet2"),
                      (self.policy_loss_history, "policy_net"),
                      (total_loss_history,       "total")]:
            show_plot_and_save(h, nn)
        
        self.policy.show_histories_for_debugs()
