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
"""

class SoftActorCriticAgent:

    def __init__(self, ):

