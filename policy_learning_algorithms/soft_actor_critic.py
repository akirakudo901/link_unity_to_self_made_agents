"""
A trial implementation of Soft-Actor Critic as documented in the spinning-up page:
https://spinningup.openai.com/en/latest/algorithms/sac.html

A discussion on the background & motivations behind the algorithm:

Our goal is to extract, from the environment, the approximated optimal policy within a set 
of parametrized (differentiable) policies Pi. Such a policy, pi*, maximizes the expected 
cumulative reward in the environment from any given state.

We know, from the policy gradient theorem, that given a policy parameterized by phi,
the gradient in cumulative rewards under this policy with respect to phi is the gradient 
of the policy with respect to phi (under an ergodic state-space and assuming that with 
regard to such marginal changes, the probability distributions for trajectories is stationary; 
that part is stil unclear to me).

Then, improving the policy towards the right direction improves the cumulative reward.

Overview of the algorithm:

Soft Actor Critic (SAC) combines an actor critic method with the soft-Q value formulation which
arises from the maximum entropy framework.

It approximates both a policy (the "actor") and a soft Q-value function for the current
policy (the "critic") using neural networks, such that ultimately, we find the optimal stochastic
policy pi* which simultaneously maximizes the expected cumulative reward and its entropy.

The two terms come into a trade-off, which balance is determined through the "temperature parameter".

The training consists of iterations between policy evaluation and policy improvement.


Policy evaluation:

This is a step where we improve our evaluation of our current policy. 
We do this by taking our approximated Q-value function closer to what we expect from the data 
collected in the environment & the currenct policy.

To do so, we can update the Q function as follows multiple times:

Q_new(s,a) <- r + gamma * V_old(s')

where V_old(s') = E_a'~pi [ Q_old(s',a') - log(pi(s',a')) ]
which can be approximated as
V_old(s') ~= Q_old(s',a') - log(pi(s',a'))
where a' is freshly samples from the current policy pi at state s'.

Since the Q-value function is approximated by a neural network, the repeated update can be
achieved through minimizing the following cost function with respect to parameters of Q and 
a buffer D of previous experiences, through gradient descent with respect to theta:

J_Q (theta) = 1/2 * E_(s,a,r,s',d)~D [ Q_theta (s,a) - y(r,s',d) ]^2
where y(r,s',d) is the target value calculated through evaluation against the neural networks.

To stabilize learning, the authors employ the clipped double-Q trick, whereby
two neural networks together approximate the Q-value functions, and update each other
using the minimum of the two networks' evaluations as target value; this way, we reduce
instability arising from initial over-estimation of Q-values being exploited by the networks.

So in essense, there are two parmeterized Q-functions theta1 and theta2, both updated 
minimizing the above loss function, where y(r,s',d) is set to be:

y(r,s',d) = r + (1-d)*E_a'~pi [ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]
 ~= r + (1-d)*[ min(Q_theta1(s', a'), Q_theta2(s', a')) - log(pi(s',a')) ] 
 where a' is sampled fresh based on policy pi.

In such a case, the gradient of J_Q with respect to theta - 1 or 2 - can be estimated as the 
following:

Delta_theta J_Q(theta) 
 ~= Delta_theta 1/2 * [ Q_theta(s,a) - r - (1-d)*[ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]^2
 where (s,a,r,s',d) are sampled from the replay buffer D
 = Delta_theta Q_theta(s,a) * [ Q_theta(s,a) - r - (1-d)*[ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]

Here min(Q_theta1(s',a'), Q_theta2(s',a')) is a constant we evaluate as target value, rather than a function
we have to obtain the gradient of with respect to theta1/2.


The authors of the original paper on SAC prove that such policy evaluation will eventually 
converge the soft-Q function to the optimal one of the corresponding policy.


Policy improvement:

This is a step where we improve our policy based on our evaluation of how good it currently is.
Through policy evaluation, we have a Q-value function with better accuracy than before.
This allows us to improve our policy by adjusting probabilities of actions given state in the 
policy to more closely match what is in the Q-value function.

Specificially, as our policy is approximated by a neural network with parameter phi, our goal
is to find, within the parameter space of phi, the optimal (or simply a better) policy.
This can be achieved, as proven by the authors of the original SAC paper, by finding a policy 
which minimizes the distance between it and the normalized exponent of the Q-value function 
distribution for different actions (which is equivalent to finding the projection from the 
normalized and exponentiated Q-value function distribution's of actions to the parameter space),
the "distance" specifically measured by KL divergence.

Here, in mathematical notation, we are finding a policy pi', such that
pi' = argmin_Pi E_s~D [ DKL(pi'(s,.) || exp(Q_theta(s,.)) / Z_theta(s) ) ]
where Z_theta(s) normalizes its numerator into a probability, and a can be freshly sampled
from the policy for estimation.

Since all policies are parameterized by phi, we can adjust phi according to minimizing the loss function
J_pi(phi) 
 = E_s~D[ DKL(pi_phi(s,.) || exp(Q_theta(s,.)) / Z_theta(s)) ]
 = E_s~D[ pi_phi(s,.)*log{ pi_phi(s,.) / exp(Q_theta(s,.)) * Z_theta(s)} ]
 = E_s~D[ pi_phi(s,.)*{ log(pi_phi(s,.)) - Q_theta(s,.) + log(Z_theta(s)) } ]

As we are looking to minimize this with respect to phi, and theta is stationary through this
process, log(Z_theta(s)) can be ultimately ignored - convenient as it is computationally costly.
Then, the loss function can be formulated, omitting that term, as:
J_pi(phi) = E_s~D[ pi_phi(s,.)*{ log(pi_phi(s,.)) - Q_theta(s,.) }]
 *I believe Q_theta(s,.) still has to remain in the terms, since . is distributed along pi_phi.

Estimating the gradient of J_pi(phi) with respect to phi can be done in multiple ways:

1: using likelihood ratio gradient.
In this soft-Q function's case, I can't quite feel confident to use this way;
but in general, we can use likelihood ratios to bypass integrals and make estimation
through sampling. 

2: using the reparameterization trick.
This approach regards the stochastic policy, determining the distribution of actions in the
above loss function, as composed of a noise random variable which is reshaped to approximate
the optimal policy through a deterministic tranformation. Doing so allows us to take the 
derivative of the deterministic transformation with respect to phi, thus creating an estimate of
the loss function's gradient which generally has been found to have lower variance compared to
that obtained through likelihood ratio gradient.

Let a noise value epsilon (eps) be sampled from a random variable q(eps), such as a spherical 
gaussian distribution. Also let the action in the policy be obtained by applying a transformation
f_phi(x|s) which is parameterized by phi, to eps, such that pi_phi(a|s) = f_phi(q(eps)|s).
Then, we can rewrite the above loss function as follows:

J_pi(phi) = E_s~D[ { pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } } ]
 = E_s~D[ E_a~pi_phi(s,.) { log(pi_phi(s,a)) - Q_theta(s,a) } ]
 = E_s~D[ E_eps~q(eps) { log(pi_phi(s, f(eps|s)) - Q_theta(s, f(eps|s) ) } ]

Then, this function's gradient with respect to phi can be obtained as follows:
Delta_phi J_pi(phi)
 = E_s~D[ E_eps~q(eps) Delta_phi { log(pi_phi(s, f(eps|s)) - Q_theta(s, f(eps|s) ) } ]
 = E_s~D[ E_eps~q(eps) Delta_phi { log(pi_phi(s, f(eps|s)) } + 
  Delta_f { log(pi_phi(s, f(eps|s) ) - Q_theta(s, f(eps|s)) } * Delta_phi { f(eps|s) } ]
 ~= Delta_phi { log(pi_phi(s, f(eps|s) )) } + 
  Delta_f { log(pi_phi(s, f(eps|s) ) - Q_theta(s, f(eps|s)) } * Delta_phi { f(eps|s) } ]
  where s is sampled from D, eps from its distribution q(eps), and f(eps|s) is an evaluated value.

In the original SAC paper, the authors use a spherical Gaussian distribution for noise vectors, 
and propose the usage of a Gaussian distribution parameterized by neural networks to form the 
policy. Then, our policy can be set to be deterministic for inference, with the chosen action
set to the mean of the learned Gaussian distribution during training.

Enough reflection, let's try implementing our algorithm.

*This file is not tested as it is just implemented for fun; once implemented and if it works well, I 
will just use open sourced implementations from there.
"""

import random

import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from policy_learning_algorithms.policy_learning_algorithm import OffPolicyLearningAlgorithm
from trainers.unityenv_base_trainer import Buffer

class SoftActorCritic(OffPolicyLearningAlgorithm):
    
    class QNet(nn.Module):
        """
        A neural network approximating the q-value function.
        """
        
        def __init__(self):
            super(SoftActorCritic.QNet, self).__init__()
            pass
        
        # def forward(self, obs : torch.tensor, actions : torch.tensor):
        #     return torch.mean(obs) + actions #TODO placeholder for debug
        
        def __call__(self, obs : torch.tensor, actions : torch.tensor):
            return torch.mean(obs, dim=1, keepdim=True) + actions #TODO placeholder for debug

    class Policy(nn.Module):
        """
        A neural network approximating the policy.
        Composed of a stochastic noise component and a deterministic component
        which maps the noises to corresponding actions.
        """
        """
        We will start by modeling the noise & deterministic components according
        to the original SAC paper and some imagination of my own:
        - noise as sampled from a multivariate gaussian distribution
        - deterministic component as a combination of tanh and the noise transformed
         based on mean and sd given by evaluating the neural network at observation
        """

        def __init__(self, observation_size : int, action_size : int):
            """
            Implements the neural network which maps observations to two values
            determining the corresponding distribution for the policy:
            - mean of approximated function
            - sd of approximated function

            :param int observation_size: The size of the observation input vector.
            :param int action_size: The size of the action vectors.
            """
            super(SoftActorCritic.Policy, self).__init__()
            fc1_out = 32
            fc2_out = 64
            self.fc1 = nn.Linear(observation_size, fc1_out)
            self.fc2 = nn.Linear(fc1_out, fc2_out)
            self.mean_layer = nn.Linear(fc2_out, action_size)
            self.sd_layer = nn.Linear(fc2_out, action_size)

            self.linear_sigmoid_stack = nn.Sequential(
                self.fc1,
                nn.Sigmoid(),
                self.fc2,
                nn.Sigmoid()
            )
            #TODO Spinning up uses more (I think?) layers. Might wanna try that out?

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
            are used as estimates of the policy at the corresponding states.

            if deterministic is False:
            :return torch.tensor squashed: The actions sampled from the policy and squashed between
            -1 and 1 using the tanh (as done with the original paper to limit possible action range)
            :return torch.tensor log_probs: The log probability for corresponding actions in squashed.
            """

            def correct_for_squash(before : torch.tensor):
                """
                Corrects for the log probabilities following squashing
                using tanh.

                :param torch.tensor before: The log probabilities to be corrected.
                :return torch.tensor after: The log probabilities after being corrected.
                """
                after = before #TODO UNDERSTAND AND APPLY THE RIGHT FORMULA HERE
                return after

            # we obtain the mean myu and sd sigma of the gaussian distribution
            stack_out = self.linear_sigmoid_stack(obs)
            myus = self.mean_layer(stack_out)
            sigmas = self.sd_layer(stack_out)
            
            # if deterministic (while in inference), return the mean of distributions
            # corresponding to actions at time of inference
            if deterministic: return myus
            
            # then evaluate the probability that action is chosen under the distribution
            dist = Normal(loc=myus, scale=sigmas) #CHANGES THIS - WILL BE MULTIVARIATE NORMAL
            actions = dist.sample(sample_shape=(num_samples, ))
            squashed = torch.tanh(actions)
            log_probs = correct_for_squash(
                dist.log_prob(actions)
                )
 
            return squashed, log_probs

    def __init__(self, 
                 learning_rate : float, 
                 discount : float, 
                 temperature : float,
                 observation_size : int,
                 action_size : int
                 ):
        self.l_r = learning_rate
        self.d_r = discount
        self.alpha = temperature

        self.qnet1 = SoftActorCritic.QNet()
        self.qnet2 = SoftActorCritic.QNet()

        self.policy = SoftActorCritic.Policy(observation_size=observation_size, action_size=action_size)
      
    def update(self, experiences : Buffer):
        # "experiences" is a list of experiences: (obs, action, reward, done, next_obs)
        POLICY_EVAL_NUM_EPOCHS = 1
        BATCH_SIZE = 8
        FRESH_ACTION_SAMPLE_SIZE = 8
        
        random.shuffle(experiences)

        observations = torch.from_numpy(np.stack([exp.obs for exp in experiences]))
        actions = torch.from_numpy(np.stack([exp.action for exp in experiences]))
        rewards = torch.from_numpy(np.stack([exp.reward for exp in experiences]))
        dones = torch.from_numpy(np.stack([exp.done for exp in experiences]))
        next_observations = torch.from_numpy(np.stack([exp.next_obs for exp in experiences]))
        # freshly sample new actions in the current policy for each observations
        # for now, we will sample FRESH_ACTION_SAMPLE_SIZE
        # TODO WHAT IS THE BEST WAY TO FRESHLY SAMPLE THOSE?
        fresh_action_samples, fresh_log_probs = self.policy(observations, FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        print("fresh_action_samples: ", fresh_action_samples, "\n")
        print("fresh_log_probs: ", fresh_log_probs, "\n")

        # 1 - policy evaluation
        for _ in range(POLICY_EVAL_NUM_EPOCHS):
            for i in range(len(experiences) // BATCH_SIZE):
                batch_obs = observations[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # print(batch_obs, "\n")
                batch_actions = actions[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # print(batch_actions, "\n")
                batch_rewards = rewards[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # print(batch_rewards, "\n")
                batch_dones = dones[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # print(batch_dones, "\n")
                batch_nextobs = next_observations[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # print(batch_nextobs, "\n")

                batch_action_samples = fresh_action_samples[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # first compute target value for all non-terminal experiences
                # print(batch_action_samples, "\n")
                
                qnet1_preds = torch.stack(
                                [self.qnet1(batch_nextobs, batch_single_sample) for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)]
                                )
                # print("qnet1_preds: ", qnet1_preds, "\n", "qnet1_shape: ", qnet1_preds.shape, "\n")
                qnet2_preds = torch.stack(
                                [self.qnet2(batch_nextobs, batch_single_sample) for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)]
                                )
                # print("qnet2_preds: ", qnet2_preds, "\n")
                minimum = torch.minimum(qnet1_preds, qnet2_preds)
                # print("minimum: ", minimum, "\n")
                mean_of_minimum = torch.mean(minimum, dim=0, dtype=torch.float32)
                # print("mean_of_minimum: ", mean_of_minimum, "\n")
                log = torch.log2(batch_action_samples)
                # print("log: ", log, "\n")
                mean_of_log = torch.mean(log)
                # print("mean_of_log: ", mean_of_log, "\n")
                targets = (batch_rewards + 
                    self.d_r * 
                    mean_of_minimum -
                    self.alpha *
                    mean_of_log
                )
                # print("targets: ", targets, "\n")


                # targets = (batch_rewards + 
                #     self.d_r * 
                #     torch.mean(
                #         torch.minimum(
                #             torch.stack(
                #                 [self.qnet1(batch_nextobs, batch_single_sample) for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)]
                #                 ),
                #             torch.stack(
                #                 [self.qnet2(batch_nextobs, batch_single_sample) for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)]
                #                 )
                #         ), dim=0, dtype=torch.float32
                #     ) -
                #     self.alpha *
                #     torch.mean(
                #         torch.log2(batch_action_samples)
                #     ) #unsure if natural log or log 2; used log 2 for now
                # )
                # mask experiences for which done is true = 1
                # then compute prediction value for all non-terminal experiences
                # finally take loss through MSELoss
                # backpropagate that loss to update q_nets
                pass

        # 2 - policy improvement

    
    def get_optimal_action(self, state):
        pass
    
    def save(self, task_name: str):
        pass
    
    def load(self, path: str):
        pass