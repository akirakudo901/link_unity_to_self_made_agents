"""
A simpler training code which proves that DDQN learns Cartpole quite well - 
which suggests that our trainer codes have bugs in them that prevent DDQN_Cartpole from learning...
"""

import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from models.policy_learning_algorithms.double_deep_q_network import DoubleDeepQNetwork
from models.trainers.utils.buffer import NdArrayBuffer


env = gym.make("CartPole-v1")#, render_mode='human')

state_size = env.observation_space.shape[0]
action_size = 2
print(f"State size: {state_size}; Action size: {action_size}.")

# reset the environment
env_info = env.reset()

ddqn = DoubleDeepQNetwork(obs_dim_size=state_size,
                          act_num_discrete=action_size,
                          l_r=1e-3,
                          d_r=0.99,
                          soft_update_coefficient=0.005,
                          update_target_every_N_updates=1,
                          dqn_layer_sizes=(64, 64), 
                          env=env)

def get_random_action():
    # sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    a = np.array(1) if random.random() >= 0.5 else np.array(0)
    return a

def train_simply(ddqn : DoubleDeepQNetwork, 
                 n_episodes : int, 
                 print_every : int, 
                 max_t : int = 1000,
                 initial_epsilon : float = 1.0,
                 min_epsilon : float = 0.1,
                 eps_decay_value : float = 0.99):
    
    def evaluate(n_samples):
        eval_scores = []

        for _ in range(n_samples):
            state, _ = env.reset()
            terminated, truncated = False, False
            score = 0

            while not (terminated or truncated):
                action = ddqn(state)

                next_state, reward, terminated, truncated, _ = env.step(action)        
                
                state = next_state
                score += reward
            
            eval_scores.append(score)
        
        print('\rEpisode {}\t Score: {:.2f}'.format(i_episode, sum(eval_scores) / len(eval_scores)))

    buffer = NdArrayBuffer(max_size=100000,
                           obs_shape=env.observation_space.shape,
                           act_shape=env.action_space.shape)
    
    eps = initial_epsilon

    scores = []
    
    for i_episode in range(1, n_episodes+1):
        
        state, _ = env.reset()
        score = 0
        
        for _ in range(max_t):

            if eps > random.random():
                action = get_random_action()
            else:
                action = np.squeeze(ddqn(np.expand_dims(state, 0)), axis=0)
   
            next_state, reward, terminated, truncated, _ = env.step(action)

            buffer.append_experience(obs=state,
                                     act=action,
                                     rew=reward,
                                     don=terminated,
                                     next_obs=next_state)
            if buffer.size() >= 10:
                ddqn.update(buffer)
            
            state = next_state
            score += reward
            
            if not eps <= min_epsilon:
                eps *= eps_decay_value
            else:
                eps = min_epsilon

            if (terminated or truncated):
                break 
        scores.append(np.mean(score))
        
        if i_episode % print_every == 0:
            evaluate(3)            

    return scores

scores = train_simply(ddqn=ddqn, 
                      n_episodes=100,
                      print_every=10,
                      initial_epsilon=0.5,
                      min_epsilon=0.1,
                      eps_decay_value=0.99)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("DDQN_simpler_training_cumulative_reward.png")
plt.show()

env.close()