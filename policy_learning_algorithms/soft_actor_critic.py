"""
A trial implementation of Soft-Actor Critic as documented in the spinning-up page:
https://spinningup.openai.com/en/latest/algorithms/sac.html

A discussion on the background & motivations behind the algorithm:

Our goal is to extract, from the environment, the approximated optimal policy pi* within a set 
of parametrized (differentiable) policies Pi. Such a policy pi* maximizes the expected 
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

where gamma is the discount rate, and 
V_old(s') = E_a'~pi [ Q_old(s',a') - log(pi(s',a')) ]
which can be approximated as
V_old(s') ~= Q_old(s',a') - log(pi(s',a'))
where a' is freshly samples from the current policy pi at state s'.

Since the Q-value function is approximated by a neural network, the repeated update can be
achieved through minimizing the following cost function with respect to the parameters theta 
of Q and a buffer D of previous experiences, through gradient descent with respect to theta:

J_Q (theta) = 1/2 * E_(s,a,r,s',d)~D [ Q_theta (s,a) - y(r,s',d) ]^2
where y(r,s',d) is the target value calculated through evaluation against the neural networks.

To stabilize learning, the authors employ the clipped double-Q trick, whereby
two neural networks together approximate the Q-value functions, and update each other
using the minimum of the two networks' evaluations as target value; this way, we reduce
instability arising from initial over-estimation of Q-values being exploited by the networks.

So in essense, there are two parmeterized Q-functions theta1 and theta2, both updated 
minimizing the above loss function, where y(r,s',d) is set to be:

y(r,s',d) = r + gamma*(1-d)*E_a'~pi [ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]
 ~= r + gamma*(1-d)*[ min(Q_theta1(s', a'), Q_theta2(s', a')) - log(pi(s',a')) ] 
 where gamma is the discount rate and a' is sampled fresh based on policy pi.

In such a case, the gradient of J_Q with respect to theta - 1 or 2 - can be estimated as the 
following:

Delta_theta J_Q(theta) 
 ~= Delta_theta 1/2 * [ Q_theta(s,a) - r - gamma*(1-d)*[ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]^2
 where (s,a,r,s',d) are sampled from the replay buffer D
 = Delta_theta Q_theta(s,a) * [ Q_theta(s,a) - r - gamma*(1-d)*[ min(Q_theta1(s',a'), Q_theta2(s',a')) - log(pi(s',a')) ]

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
 = E_s~D[ (Integrate over A) pi_phi(s,a)*log{ pi_phi(s,a) / exp(Q_theta(s,a)) * Z_theta(s)} ]
 = E_s~D[ (Integrate over A) pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) + log(Z_theta(s)) } ]
 where A is the set of all actions.
 
As we are looking to minimize this with respect to phi, and theta is stationary through this
process, log(Z_theta(s)) can be ultimately ignored as simply being a constant - convenient as 
it is computationally costly.
Then, the loss function can be formulated, omitting that term, as:
J_pi(phi) = E_s~D[ (Integrate over A) pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) }]
 
Estimating the gradient of J_pi(phi) with respect to phi can be done in multiple ways:

1: using likelihood ratio gradient.
Taking the gradient with respect to phi of the loss function above goes as follows:
Delta_phi J_pi(phi) 
 = Delta_phi E_s~D[ (Integrate over A) pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) }]
 = E_s~D[ (Integrate over A) Delta_phi ( pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } ) ]
 = E_s~D[ (Integrate over A) ( Delta_phi pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } ) + 
 ( pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } )]
 = E_s~D[ (Integrate over A) ( Delta_phi pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } ) + 
 E_a~pi_phi[ pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } ] ]

The second term of this equation can be estimated using random samples - but this is not the case
for the first term. To be able to also estimate the first term with random samples, we can rewrite
it using the REINFORCE trick:
Delta_phi pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) }
 = {Delta_phi pi_phi(s,a)} / pi_phi(s,a) * pi_phi(s,a) * { log(pi_phi(s,a)) - Q_theta(s,a) }
 = Delta_phi {log {pi_phi(s,a)} } * pi_phi(s,a) * { log(pi_phi(s,a)) - Q_theta(s,a) }
 = pi_phi(s,a) * Delta_phi {log {pi_phi(s,a)} } * { log(pi_phi(s,a)) - Q_theta(s,a) }

Substituting this back, we get:
Delta_phi J_pi(phi) 
 = E_s~D[ (Integrate over A) ( Delta_phi pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } ) + 
 E_a~pi_phi[ pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } ] ]
 = E_s~D[ 
    (Integrate over A) 
    ( pi_phi(s,a) * Delta_phi {log {pi_phi(s,a)} } * { log(pi_phi(s,a)) - Q_theta(s,a) } ) + 
 E_a~pi_phi[ pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } ] ]
 = E_s~D[
 E_a~pi_phi[ Delta_phi {log {pi_phi(s,a)} } * { log(pi_phi(s,a)) - Q_theta(s,a) } ] + 
 E_a~pi_phi[ pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } ] ]
 = E_s~D[
 E_a~pi_phi[ ( Delta_phi {log {pi_phi(s,a)} } * { log(pi_phi(s,a)) - Q_theta(s,a) } ) + 
 ( pi_phi(s,a) * Delta_phi { log(pi_phi(s,a)) } ) ] ]

 which can then be estimated by sampling according to the buffer D and policy pi_phi.

2: using the reparameterization trick.
This approach regards the stochastic policy, determining the distribution of actions in the
above loss function, as composed of a noise random variable which is reshaped to approximate
the optimal policy through a deterministic tranformation. Doing so allows us to take the 
derivative of the deterministic transformation with respect to phi, thus creating an estimate of
the loss function's gradient which generally has been found to have lower variance compared to
that obtained through likelihood ratio gradient.

Let a noise value epsilon (eps) be sampled from a random variable q(eps), such as a spherical 
gaussian distribution - this is appropriate in the case of the original SAC paper, which uses a 
multivariate normal distribution as policy approximator, which covariance matrix is diagonal. 
Also let the action in the policy be obtained by applying a transformation f_phi(x|s) which is parameterized by phi, to eps, such that a = f_phi(eps|s).
Then, we can rewrite the above loss function as follows:

J_pi(phi) = E_s~D[ E_a~pi_phi[ pi_phi(s,a)*{ log(pi_phi(s,a)) - Q_theta(s,a) } ] ]
 = E_s~D[ E_eps~q(eps)[ log(pi_phi( s, f_phi(eps|s) )) - Q_theta( s, f_phi(eps|s) ) ] ]

This allowed us to reformulate what used to be an expectation over the action distributed along
pi_phi, which we will tweak when taking the gradient with respect to phi, into an expectation 
over the action distributed along q(eps), which does not rely on phi.

Then, this function's gradient with respect to phi can be obtained as follows:
Delta_phi J_pi(phi)
 = Delta_phi E_s~D[ E_eps~q(eps)[ log(pi_phi( s, f_phi(eps|s) )) - Q_theta( s, f_phi(eps|s) ) ] ]
 = E_s~D[ E_eps~q(eps)[ Delta_phi{ log(pi_phi( s, f_phi(eps|s) )) - Q_theta( s, f_phi(eps|s) ) } ] ]
 = E_s~D[ E_eps~q(eps)[ Delta_phi{ log(pi_phi( s, f_phi(eps|s) )) }
  - Delta_f{ Q_theta( s, a ) }*Delta_phi{ f_phi(eps|s) } ] ]

Dealing with the first term Delta_phi { log(pi_phi( s, f_phi(eps|s) )) } is quite complicated, so
I will leave it to this arxiv paper: https://arxiv.org/pdf/2112.15568v1.pdf
Essentially, this part will evaluate to:
Delta_phi{ log(pi_phi( s, f_phi(eps|s) )) }
 = Delta_phi{ log(pi_phi( s, a )) } + Delta_a{ log(pi_phi( s, a )) }*Delta_phi{ f_phi(eps|s) }

The overall equation for the derivative thus evaluates to:
Delta_phi J_pi(phi)
 = E_s~D[ E_eps~q(eps)[ 
    Delta_phi{ log(pi_phi( s, a )) } + Delta_a{ log(pi_phi( s, a )) }*Delta_phi{ f_phi(eps|s) }
    - Delta_f{ Q_theta( s, a ) }*Delta_phi{ f_phi(eps|s) } 
    ] ]
    
Enough reflection, let's try implementing our algorithm.

*This file is not tested as it is just implemented for fun; once implemented and if it works well, I 
will just use open sourced implementations from there.
"""

from datetime import datetime
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from policy_learning_algorithms.policy_learning_algorithm import OffPolicyLearningAlgorithm
from trainers.unityenv_base_trainer import Buffer

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
        
        # #TODO placeholder for debug
        # def __call__(self, obs : torch.tensor, actions : torch.tensor):
        #     return torch.mean(obs, dim=1, keepdim=True) + actions 

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

        def __init__(self, observation_size : int, action_size : int):
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
            """
            super(SoftActorCritic.Policy, self).__init__()
            fc_out = 256
            self.fc1 = nn.Linear(observation_size, fc_out)
            self.fc2 = nn.Linear(fc_out, fc_out)
            
            self.linear_relu_stack = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU()
            )

            self.mean_layer = nn.Linear(fc_out, action_size)
            self.sd_layer = nn.Linear(fc_out, action_size)

        def forward(self, 
                    obs : torch.tensor,
                    num_samples : int = 1,
                    deterministic : bool = False,
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
            
            # we squash action values to be between -1 and 1 using tanh
            squashing_function = torch.tanh

            # we obtain the mean myu and sd sigma of the gaussian distribution
            stack_out = self.linear_relu_stack(obs)
            myus = self.mean_layer(stack_out)
            sigmas = torch.abs(self.sd_layer(stack_out))
            # print("myus: ", myus, "myus.shape: ", myus.shape)
            # print("sigmas: ", sigmas, "sigmas.shape: ", sigmas.shape)
            
            # if deterministic (while in inference), return the mean of distributions
            # corresponding to actions at time of inference, but squashed as needed
            if deterministic: return squashing_function(myus)
            
            # then evaluate the probability that action is chosen under the distribution
            dist = Normal(loc=myus, scale=sigmas) #MultivariateNormal with diagonal covariance
            actions_num_samples_first = dist.sample(sample_shape=(num_samples, )).to(SoftActorCritic.NUMBER_DTYPE)
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
            log_probs = SoftActorCritic.Policy._correct_for_squash(
                before_correction, actions
                )
            
            return squashed, log_probs
        
        def _correct_for_squash(before : torch.tensor, actions : torch.tensor):
            """
            Corrects for the log probabilities following squashing
            using tanh. 

            Given before is a log probability and the transformation is elementwise tanh,
            the adjustment is 
                [before - Sum(1 - tanh^2(u_i))] 
                where u_i is the ith element in a single action.
                This is since
                1) Correction involves multiplying the original probability with the determinant
                    of the Jacobian of the transformation.
                2) The Jacobian of elementwise tanh is a diagonal, with each entry being each 
                    element of each action transformed by the derivative of tanh, which is 
                    (1 - tanh^2(x)). We divide by this factor here as after results from tanh 
                    correction of before.
                3) Since we are dealing with log probabilities, multiplication turn to addition.

            :param torch.tensor before: The log probabilities to be corrected.
            Will be of the form [batch_size, num_samples].
            :param torch.tensor actions: The actions which log probabilities are in question.
            Will be of the form [batch_size, num_samples, action_size].
            :return torch.tensor after: The log probabilities after being corrected.
            Will be of the form [batch_size, num_samples].
            """
            # compute the trace of the Jacobian of tanh squashing for each action
            # print("before: ", before, "before.shape: ", before.shape)
            # print("actions: ", actions, "actions.shape: ", actions.shape)
            jacobian_trace = torch.sum((1 - torch.tanh(actions)**2), dim=2)
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
                 observation_size : int,
                 action_size : int,
                 update_qnet_every_N_gradient_steps : int = 1000,
                 optimizer : optim.Optimizer = optim.Adam, 
                 device = None
                 ):
        self.q_net_l_r = q_net_learning_rate
        self.pol_l_r = policy_learning_rate
        self.d_r = discount
        self.alpha = temperature
        self.update_qnet_every_N_gradient_steps = update_qnet_every_N_gradient_steps
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Device was not specified, so using:', self.device, '.')
        elif device == 'cpu':
            self.device = torch.device('cpu')
            print("Device specified as cpu; using: ", self.device, ".")
        else:
            self.device = device
            print('Using:', self.device, ' as specified.')

        self.qnet_update_counter = 1

        # two q-functions are used to approximate values during training
        self.qnet1 = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.qnet2 = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.optim_qnet1 = optimizer(self.qnet1.parameters(), lr=q_net_learning_rate)
        self.optim_qnet2 = optimizer(self.qnet2.parameters(), lr=q_net_learning_rate)
        # two target networks are updated less (e.g. every 1000 steps) to compute targets for q-function update
        self.qnet1_tar = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        self.qnet2_tar = SoftActorCritic.QNet(observation_size=observation_size, action_size=action_size).to(self.device)
        # transfer the weights to sync
        self._update_target_networks()
        
        # initialize policy 
        self.policy = SoftActorCritic.Policy(observation_size=observation_size, action_size=action_size).to(self.device)
        self.optim_policy = optimizer(self.policy.parameters(), lr=policy_learning_rate)
    
    def __call__(self, state):
        """
        A format put in place where SAC is called raw to obtain actions from it.

        :param state: The state of observation in question to which this algorithm was applied.
        Could be a numpy array or torch tensor? TODO ASCERTAIN!
        """
        return self.get_optimal_action(state)
      
    def update(self, experiences : Buffer):
        # "experiences" is a list of experiences: (obs, action, reward, done, next_obs)
        POLICY_EVAL_NUM_EPOCHS = 500
        POL_EVAL_BATCH_SIZE = 1028
        POL_EVAL_FRESH_ACTION_SAMPLE_SIZE = 1

        POLICY_IMP_NUM_EPOCHS = 500
        POL_IMP_BATCH_SIZE = 1028
        POL_IMP_FRESH_ACTION_SAMPLE_SIZE = 1
        random.shuffle(experiences)
        observations, actions, rewards, dones, next_observations = SoftActorCritic._unzip_experiences(experiences, device=self.device)
        # print("observations.shape: ", observations.shape)
        # print("actions.shape: ", actions.shape)
        # print("rewards.shape: ", rewards.shape)
        # print("dones.shape: ", dones.shape)
        # print("next_observations.shape: ", next_observations.shape)
        
        # freshly sample new actions in the current policy for each observations
        # for now, we will sample FRESH_ACTION_SAMPLE_SIZE
        # TODO WHAT IS THE BEST WAY TO FRESHLY SAMPLE THOSE?
        fresh_action_samples, fresh_log_probs = self.policy(observations, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        # print("fresh_action_samples: ", fresh_action_samples, "\n")
        # print("fresh_action_samples.shape: ", fresh_action_samples.shape, "\n")
        # print("fresh_log_probs: ", fresh_log_probs, "\n")
        # print("fresh_log_probs.shape: ", fresh_log_probs.shape, "\n")

        # 1 - policy evaluation
        for _ in range(POLICY_EVAL_NUM_EPOCHS):
            for i in range(len(experiences) // POL_EVAL_BATCH_SIZE + 1):
                batch_start = i*POL_EVAL_BATCH_SIZE
                batch_end = min((i+1)*POL_EVAL_BATCH_SIZE, len(experiences))

                batch_obs = observations[batch_start : batch_end].detach()
                # print(batch_obs, "\n")
                # print("batch_obs.shape: ", batch_obs.shape, "\n")
                batch_actions = actions[batch_start : batch_end].detach()
                # print(batch_actions, "\n")
                # print("batch_actions.shape: ", batch_actions.shape, "\n")
                batch_rewards = rewards[batch_start : batch_end].detach()
                # print(batch_rewards, "\n")
                # print("batch_rewards.shape: ", batch_rewards.shape, "\n")
                batch_dones = dones[batch_start : batch_end].detach()
                # print(batch_dones, "\n")
                # print("batch_dones.shape: ", batch_dones.shape, "\n")
                batch_nextobs = next_observations[batch_start : batch_end].detach()
                # print(batch_nextobs, "\n")
                # print("batch_nextobs.shape: ", batch_nextobs.shape, "\n")

                # batch_action_samples' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE, action_size]
                batch_action_samples = fresh_action_samples[batch_start : batch_end].detach()
                # print("batch_action_samples: ", batch_action_samples, "\n")
                # print("batch_action_samples.shape: ", batch_action_samples.shape, "\n")
                
                # batch_log_probs' shape is [batch_size, POL_EVAL_FRESH_ACTION_SAMPLE_SIZE]
                batch_log_probs = fresh_log_probs[batch_start : batch_end].detach()
                # print("batch_log_probs: ", batch_log_probs, "\n")
                # print("batch_log_probs.shape: ", batch_log_probs.shape, "\n")
                
                # first compute target value for all experiences (terminal ones are only the rewards)
                targets = self._compute_qnet_target(batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs)
                # print("targets: ", targets, "targets.shape: ", targets.shape, "\n")

                # then compute prediction value for all experiences
                predictions1 = torch.squeeze(self.qnet1(obs=batch_obs, actions=batch_actions), dim=1)
                predictions2 = torch.squeeze(self.qnet2(obs=batch_obs, actions=batch_actions), dim=1)
                # print("predictions1: ", predictions1, "predictions1.shape: ", predictions1.shape)
                
                # finally take loss through MSELoss
                criterion = nn.MSELoss()
                loss1 = criterion(predictions1, targets)
                loss2 = criterion(predictions2, targets)
                # backpropagate that loss to update q_nets
                self.optim_qnet1.zero_grad()
                loss1.backward(retain_graph=True)
                self.optim_qnet1.step()
                
                self.optim_qnet2.zero_grad()
                loss2.backward()
                self.optim_qnet2.step()
                
                # finally increment the update counter and increment
                # update the target network every N gradient steps
                if self.qnet_update_counter % self.update_qnet_every_N_gradient_steps == 0:
                    self._update_target_networks()
                
        # 2 - policy improvement
        # at this point, the q-network weights are adjusted to reflect the q-value of
        # the current policy. We just have to take a gradient step with respect to the  
        # distance between this q-value and the current policy
        
        # in order to estimate the gradient, we again sample some actions at this point in time.
        # TODO COULD WE USE ACTIONS SAMPLED BEFORE WHICH WERE USED FOR Q-NETWORK UPDATE? NOT SURE
        fresh_action_samples2, fresh_log_probs2 = self.policy(observations, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, deterministic=False)
        
        for _ in range(POLICY_IMP_NUM_EPOCHS):
            for i in range(len(experiences) // POL_IMP_BATCH_SIZE + 1):
                batch_start = i*POL_IMP_BATCH_SIZE
                batch_end = min((i+1)*POL_IMP_BATCH_SIZE, len(experiences))

                batch_obs = observations[batch_start : batch_end].detach()
                # print(batch_obs, "\n")
                
                # batch_action_samples' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE, action_size]
                batch_action_samples = fresh_action_samples2[batch_start : batch_end].detach()
                # print("batch_action_samples: ", batch_action_samples, "\n")
                # print("batch_action_samples.size: ", batch_action_samples.size, "\n")
                
                # batch_log_probs' shape is [batch_size, POL_IMP_FRESH_ACTION_SAMPLE_SIZE]
                batch_log_probs = fresh_log_probs2[batch_start : batch_end].detach()
                # print("batch_log_probs: ", batch_log_probs, "\n")
                # print("batch_log_probs.size: ", batch_log_probs.size, "\n")
                
                # Then, we compute the loss function: which relates the exponentiated distribution of
                # the q-value function with the current policy's distribution, through KL divergence
                exponentiated_qval = self._compute_exponentiated_qval(batch_obs, batch_action_samples)
                policy_val = torch.mean(batch_log_probs, dim=1)
                criterion = nn.KLDivLoss(reduction="batchmean")
                loss = criterion(policy_val, exponentiated_qval)
                
                # then, we improve the policy by minimizing this loss 
                self.optim_policy.zero_grad()
                loss.backward()
                self.optim_policy.step()
                
    def _compute_exponentiated_qval(self, batch_obs, batch_action_samples):
        # predicted q_val for each qnet of the shape [batch, num_samples]
        qnet1_exp_val = torch.cat([
                                    self.qnet1(batch_obs, torch.squeeze(batch_action_samples[:, i, :], dim=1))
                                    for i in range(batch_action_samples.shape[1])  
                                    # self.qnet1(batch_obs, torch.squeeze(batch_single_sample, dim=1)) 
                                    # for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)
                                    ],
                                    dim=1
                                )
        qnet2_exp_val = torch.cat([
                                    self.qnet2(batch_obs, torch.squeeze(batch_action_samples[:, i, :], dim=1))
                                    for i in range(batch_action_samples.shape[1])  
                                    # self.qnet2(batch_obs, torch.squeeze(batch_single_sample, dim=1)) 
                                    # for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)
                                    ],
                                    dim=1
                                )
        # print("qnet1_exp_val: ", qnet1_exp_val, "\n", "qnet1_exp_val.shape: ", qnet1_exp_val.shape, "\n")
        # the minimum of those predictions of the shape [batch, num_samples]
        exp_qval_minimum = torch.minimum(qnet1_exp_val, qnet2_exp_val)
        # print("exp_qval_minimum: ", exp_qval_minimum, "\n", "exp_qval_minimum.shape: ", exp_qval_minimum.shape, "\n")
        # the mean of those predictions of the shape [batch]
        mean_exp_qval = torch.mean(exp_qval_minimum, dim=1, dtype=torch.float32)
        # print("mean_exp_qval: ", mean_exp_qval, "\n", "mean_exp_qval.shape: ", mean_exp_qval.shape, "\n")
        return mean_exp_qval

    def _compute_qnet_target(self, batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs):
        qnet1_tar_preds = torch.cat(
                                [
                                    self.qnet1_tar(batch_nextobs, torch.squeeze(batch_action_samples[:, i, :], dim=1))
                                    for i in range(batch_action_samples.shape[1])  
                                    # self.qnet1_tar(batch_nextobs, torch.squeeze(batch_single_sample, dim=1)) 
                                    # for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)
                                ], 
                                dim=1
                                )
        # print("qnet1_tar_preds: ", qnet1_tar_preds, "\n", "qnet1_tar_preds.shape: ", qnet1_tar_preds.shape, "\n")
        qnet2_tar_preds = torch.cat(
                                [
                                    self.qnet2_tar(batch_nextobs, torch.squeeze(batch_action_samples[:, i, :], dim=1))
                                    for i in range(batch_action_samples.shape[1])  
                                    # self.qnet2_tar(batch_nextobs, torch.squeeze(batch_single_sample, dim=1)) 
                                    # for batch_single_sample in torch.split(batch_action_samples, 1, dim=1)
                                ], 
                                dim=1
                                )
        # print("qnet2_tar_preds: ", qnet2_tar_preds, "\n")
        minimum = torch.minimum(qnet1_tar_preds, qnet2_tar_preds)
        # print("minimum: ", minimum, "\n")
        mean_of_minimum = torch.mean(minimum, dim=1, dtype=SoftActorCritic.NUMBER_DTYPE)
        # print("mean_of_minimum: ", mean_of_minimum, "\n")
        mean_of_log = torch.mean(batch_log_probs, dim=1)
        # print("mean_of_log: ", mean_of_log, "\n")
        targets = (
                    batch_rewards + 
                    self.d_r *
                    (1.0 - batch_dones.to(SoftActorCritic.NUMBER_DTYPE)) *
                    (
                        mean_of_minimum -
                        self.alpha *
                        mean_of_log
                    )
                )
        # print("targets: ", targets, "\n")

        return targets
    
    def _update_target_networks(self):
        """
        Updates the target networks to hold values from the 
        correspondingly q-networks.
        """
        self.qnet1_tar.load_state_dict(self.qnet1.state_dict())
        self.qnet2_tar.load_state_dict(self.qnet2.state_dict())

    def _unzip_experiences(experiences : Buffer, device = None):
        """
        Unzips the experiences into groups of observations, actions, rewards, 
        done flags, and next observations, to be returned.

        :param Buffer experiences: A Buffer containing obs, action, reward, done and next_obs.
        :return Tuple[torch.tensor]: Tensors of each component in the Buffer; 
        observations, actions, rewards, dones, next_observations.
        """
        if device != None:
            observations      = torch.from_numpy(np.stack([exp.obs      for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE).to(device)
            actions           = torch.from_numpy(np.stack([exp.action   for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE).to(device)
            rewards           = torch.from_numpy(np.stack([exp.reward   for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE).to(device)
            dones             = torch.from_numpy(np.stack([exp.done     for exp in experiences])).to(device)
            next_observations = torch.from_numpy(np.stack([exp.next_obs for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE).to(device)
        else:
            observations      = torch.from_numpy(np.stack([exp.obs      for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE)
            actions           = torch.from_numpy(np.stack([exp.action   for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE)
            rewards           = torch.from_numpy(np.stack([exp.reward   for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE)
            dones             = torch.from_numpy(np.stack([exp.done     for exp in experiences]))
            next_observations = torch.from_numpy(np.stack([exp.next_obs for exp in experiences])).to(SoftActorCritic.NUMBER_DTYPE)
        return observations,actions,rewards,dones,next_observations
    
    def get_optimal_action(self, state):
        """
        Computes the currently optimal action given an observation state.
        State can be either a torch tensor or numpy array, both being converted
        into a torch tensor before further processing is done.

        :param state: The observation state given as torch.tensor or np.ndarray.
        :return torch.tensor action: The action the policy deems optimal. 
        """
        # if type is not torch.tensor, try casting
        if type(state) != type(torch.tensor([0])):
            try:
                state = torch.from_numpy(state)
            except:
                raise Exception("Error in reading observation within SAC get_optimal_action; \
                                'state' needs to be one of torch.tensor or np.ndarray.")
        action = self.policy(obs=state.to(self.device), deterministic=True)
        return action
    
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