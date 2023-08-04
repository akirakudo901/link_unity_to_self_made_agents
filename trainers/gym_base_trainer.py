"""
Represents a base trainer which can be adapted to any gym environment
by making a trainer which inherits from it.

~Classes~
 BaseTrainer: a handler of interaction between the learning algorithm and the gym environment
              to collect experience and update the algorithm
"""

import random
from typing import List, NamedTuple, Tuple
 
import gymnasium
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]


class GymOnPolicyBaseTrainer:
    pass
    
    # def __init__(
    #         self, 
    #         env,
    #         learning_algorithm
    #     ):
    #     """
    #     Creates an on policy base trainer with a given gym environment and a 
    #     learning algorithm which holds a policy.
    #     Enables the generation of the experience for a single time step and return it
    #     to be used for on-policy learning.

    #     :param env: The gymnasium environment used.
    #     :param learning_algorithm: The algorithm which provides policy, evaluating 
    #     actions given states per batches.
    #     """
    #     self.env = env
    #     self.learning_algorithm = learning_algorithm

    #     self.cumulative_reward : float = 0.0
    #     self.last_action : np.ndarray = None
    #     self.last_observation : np.ndarray = None

    # def generate_experience(
    #     self,
    #     exploration_function,
    #     ) -> Tuple[Buffer, List[Trajectory], List[float]]:
    #     """
    #     Executes a single step in the environment for the agent, storing the last state & action 
    #     and cumulative reward. 
    #     Returns the experience of the agent in the last step, and trajectories & cumulative rewards
    #     for agents who reached a terminal step.
    #     The experience involves some exploration (vs. exploitation) which behavior 
    #     is determined by exploration_function.
    #     :param exploration_function: A function which controls how exploration is handled 
    #     during collection of experiences.
    #     :returns Buffer experiences: The experiences of every agent that took action in
    #     the last step.
    #     :returns List[Trajectory] terminal_trajectories: The list of trajectories for agents
    #     that reached a terminal state.
    #     :returns List[float] terminal_cumulative_rewards: The list of rewards for agents that 
    #     reached a terminal state. 
    #     """
    #     experiences : Buffer = []
    #     terminal_trajectories : List[Trajectory] = []
    #     terminal_cumulative_rewards : List[float] = []
    #     """
    #     Generating one batch of trajectory comes in the following loop
    #     1) Get info from environment via step(action)
    #      decision_steps for agents not in terminal state, as list of:
    #      - an observation of the last steps
    #      - the reward as its result in float
    #      - an int of the agent's id as unique identifier
    #      - an action_mask, only available for multi-discrete actions

    #      terminal_steps for agents who reached a terminal state, as list of:
    #      - an observation of the last steps
    #      - the reward as its result in float
    #      - "interrupted" as a boolean if the agent was interrupted in the last step
    #      - an int of the agent's id as unique identifier
    #     """
    #     decision_steps, terminal_steps = self.env.step(behavior_name)
        
    #     # for every agent who requires a new decision
    #     for decision_agent_id in decision_steps.agent_id:
            
    #         dec_agent_idx = decision_steps.agent_id_to_index[decision_agent_id]   

    #         # if agent has no past observation
    #         if decision_agent_id not in self.trajectory_by_agent.keys():
    #             self.trajectory_by_agent[decision_agent_id] = []
    #             self.cumulative_reward_by_agent[decision_agent_id] = 0
            
    #         # if agent has already had past observation
    #         else:                
    #             # create a new experience based on the last stored info and the new info
    #             new_experience = Experience(
    #                 obs=self.last_observation_by_agent[decision_agent_id].copy(), 
    #                 action=self.last_action_by_agent[decision_agent_id].copy(), 
    #                 reward=decision_steps.reward[dec_agent_idx].copy(),
    #                 done=False,
    #                 next_obs=decision_steps.obs[0][dec_agent_idx].copy()
    #             )
    #             # add to the tuple of experiences to be returned
    #             experiences.append(new_experience)
    #             # add to trajectory and update cumulative reward
    #             self.trajectory_by_agent[decision_agent_id].append(new_experience)
    #             self.cumulative_reward_by_agent[decision_agent_id] += decision_steps.reward[dec_agent_idx]
    #         # update last observation
    #         self.last_observation_by_agent[decision_agent_id] = decision_steps.obs[0][dec_agent_idx].copy()

    #     # for every agent which has reached a terminal state
    #     for terminal_agent_id in terminal_steps.agent_id:

    #         term_agent_idx = terminal_steps.agent_id_to_index[terminal_agent_id]

    #         # if agent has no past observation, policy is not used in anyway, so we pass
    #         if terminal_agent_id not in self.trajectory_by_agent.keys(): pass

    #         # if agent has past observation
    #         else:
    #             # create the last experience based on the last stored info and the new info
    #             last_experience = Experience(
    #                 obs=self.last_observation_by_agent[terminal_agent_id].copy(), 
    #                 action=self.last_action_by_agent[terminal_agent_id].copy(), 
    #                 reward=terminal_steps.reward[term_agent_idx],
    #                 done=not terminal_steps.interrupted[term_agent_idx],
    #                 next_obs=terminal_steps.obs[0][term_agent_idx].copy()
    #             )
    #             # add to the tuple for experiences to be returned
    #             experiences.append(last_experience)
    #             # update trajectory and cumulative reward
    #             self.trajectory_by_agent[terminal_agent_id].append(last_experience)
    #             self.cumulative_reward_by_agent[terminal_agent_id] += terminal_steps.reward[term_agent_idx]
    #             # move trajectory to buffer and report cumulative reward while removing them from dict
    #             terminal_trajectories.append(self.trajectory_by_agent.pop(terminal_agent_id))
    #             terminal_cumulative_rewards.append(self.cumulative_reward_by_agent.pop(terminal_agent_id))
    #             # remove last action and observation from dicts
    #             self.last_observation_by_agent.pop(terminal_agent_id)
    #             self.last_action_by_agent.pop(terminal_agent_id)

    #     # if there still are agents requesting new actions
    #     if len(decision_steps.obs[0]) != 0:
    #         # Generate actions for agents while applying the exploration function to 
    #         # promote exploration of the world
            
    #         best_actions = exploration_function(
    #             self.learning_algorithm(decision_steps.obs[0]), 
    #             self.env
    #         )
            
    #         # Store info of action picked to generate new experience in the next loop
    #         for agent_idx, agent_id in enumerate(decision_steps.agent_id):
    #             self.last_action_by_agent[agent_id] = best_actions[agent_idx]
    #         # Set the actions in the environment
    #         # Unity Environments expect ActionTuple instances.
            
    #         action_tuple = ActionTuple()
    #         if self.env.behavior_specs[behavior_name].action_spec.is_continuous:
                
    #             action_tuple.add_continuous(best_actions)
    #         else:
    #             action_tuple.add_discrete(best_actions)
    #         self.env.set_actions(behavior_name, action_tuple)
            
    #     # Perform a step in the simulation
    #     self.env.step()

    #     return experiences, terminal_trajectories, terminal_cumulative_rewards
    
    # # TODO A function which trains the given algorithm through on-policy learning.
    # def train(self):
    #     return


class GymOffPolicyBaseTrainer:

    def __init__(
            self, 
            env,
            learning_algorithm
        ):
        """
        Creates an on policy base trainer with a given gym environment and a 
        learning algorithm which holds a policy.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param env: The gymnasium environment used.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        """
        self.env = env
        self.learning_algorithm = learning_algorithm
        
        self.env_eval = gymnasium.make(self.env.spec.id, render_mode="human")

        self.reset()
    
    def reset(self):
        """
        Reset the environment and internal variables.
        """
        self.last_action : np.ndarray = None
        self.last_observation : np.ndarray = None
        self.reset = True

    def generate_experience(
            self,
            exploration_function,
            ):
        """
        Executes a single step in the environment for the agent, storing the last state & action 
        and cumulative reward. 
        Returns the experience of the agent in the last step, and trajectories & cumulative rewards
        for agents who reached a terminal step.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :returns Experience experience: The experience of the agent as it took action in
        the last step.
        """
        
        """
        Generating a single experience comes in the following loop
        1) Get info from environment via env.step(self.last_action) as list of:
         - the resulting state in np.ndarray
         - the reward as result of the action in float
         - the boolean "terminated" determining if we reached a terminal state
         - the boolean "truncated" determining if training ended from conditions
           determined outside the environment
         - "info" which provides additional information
        """
        # if the environment is reset (beginning of training), we reset env and take a step first
        if self.reset:
            self.last_observation, _ = self.env.reset()
            self.last_action = torch.squeeze( #adjust, as output from learning_algo always has a batch dimension
                self.learning_algorithm(np.expand_dims(self.last_observation, 0)).cpu().detach()
                ).numpy()

        new_observation, reward, terminated, truncated, _ = self.env.step(self.last_action)
            
        # create a new experience based on the last stored info and the new info
        new_experience = Experience(
            obs=self.last_observation, 
            action=self.last_action, 
            reward=reward,
            done=terminated, #if terminated, we reached a terminal state
            next_obs=new_observation
        )
        # update last observation
        self.last_observation = new_observation

        # if the environment hasn't ended
        if not (terminated or truncated):
            # Generate actions for agents while applying the exploration function to 
            # promote exploration of the world
            
            best_action = exploration_function(
                torch.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    self.learning_algorithm(np.expand_dims(self.last_observation, 0)).cpu().detach()
                ).numpy(),
                self.env
            )
            
            # Store info of action picked to generate new experience in the next loop
            self.last_action = best_action
        else:
            # reset the state of this trainer (not the environment!)
            self.reset()

        return new_experience        

    def generate_batch_of_experiences(
            self, 
            buffer_size : int,
            exploration_function,
        ) -> Tuple[Buffer, float]:
        """
        Generates and returns a buffer containing "buffer_size" random experiences 
        sampled from running the policy in the environment.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.

        :param int buffer_size: The size of the buffer to be returned.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experience.
        :returns Buffer buffer: The buffer containing buffer_size experiences.
        """

        # Create an empty buffer
        buffer : Buffer = []

        # While we have not enough experience in buffer, generate experience
        while len(buffer) < buffer_size:
            new_experience = self.generate_experience(exploration_function=exploration_function)
            
            buffer.append(new_experience)
            
        # Cut down number of exps to buffer_size - useful if we generate multiple exps in each loop
        buffer = buffer[:buffer_size]

        return buffer
    
    def train(
            self,
            num_training_steps : int,
            num_new_experience : int,
            max_buffer_size : int,
            num_initial_experiences : int,
            exploration_function,
            save_after_training : bool,
            task_name : str
        ):
        """
        Trains the learning algorithm in the environment with given specifications, returning 
        the trained algorithm.

        :param int num_training_steps: The number of steps we proceed to train the algorithm.
        :param int num_new_experience: The number of new experience we collect minimum before
        every update for our algorithm.
        :param int max_buffer_size: The maximum number of experience to be kept in the buffer.
        :param int num_initial_experiences: The number of experiences to be collected in the 
        buffer first before doing any learning.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param bool save_after_training: Whether to save the policy after training.
        :param str task_name: The name of the task to log when saving after training.
        """
        
        # try:
        experiences : Buffer = []
        cumulative_rewards : List[float] = []

        # first populate the buffer with num_initial_experiences experiences
        print(f"Generating {num_initial_experiences} initial experiences...")
        init_exp = self.generate_batch_of_experiences(
            buffer_size=num_initial_experiences,
            exploration_function=exploration_function
        )
        print("Generation successful!")
        # TODO TOREMOVE

        print("Action generated at the very beginning!")
        obs = torch.unsqueeze(torch.from_numpy(init_exp[0].obs), dim=0).cuda()
        act = torch.tensor([[-0.1313,  0.0119, -0.3108,  0.0566],]).cuda()
        print("obs: ", obs, "obs.shape: ", obs.shape, "act: ", act, "act.shape: ", act.shape)
        pred = torch.squeeze(self.learning_algorithm.qnet1(obs=obs, actions=act), dim=1)
        print("pred: ", pred)
        
        experiences.extend(init_exp)
        random.shuffle(experiences)
        if len(experiences) > max_buffer_size:
            experiences = experiences[:max_buffer_size]
            
        # then go into the training loop
        for i in tqdm(range(num_training_steps)):
            
            new_exp = self.generate_batch_of_experiences(
                buffer_size=num_new_experience,
                exploration_function=exploration_function
            )
            
            experiences.extend(new_exp)
            random.shuffle(experiences)
            if len(experiences) > max_buffer_size:
                experiences = experiences[:max_buffer_size]
            self.learning_algorithm.update(experiences)

            # evaluate sometimes
            if i % (num_training_steps // 10) == (num_training_steps // 10 - 1):
                cumulative_reward = self.evaluate(1)
                cumulative_rewards.append(cumulative_reward)

                print(f"Prediction generated at step {i}!")
                pred = torch.squeeze(self.learning_algorithm.qnet1(obs=obs, actions=act), dim=1)
                print("pred: ", pred)
            
        if save_after_training: self.learning_algorithm.save(task_name)
        print("Training ended successfully!")

        # except KeyboardInterrupt:
        #     print("\nTraining interrupted, continue to next cell to save to save the model.")
        # finally:
        #     print("Closing envs...")
        #     self.env.close()
        #     self.env_eval.close()
        #     print("Successfully closed envs!")

        #     # Show the training graph
        #     try:
        #         plt.plot(range(len(cumulative_rewards)), cumulative_rewards)
        #         plt.savefig(f"{task_name}_cumulative_reward_fig.png")
        #         plt.show()
        #     except ValueError:
        #         print("\nPlot failed on interrupted training.")
                
        #     return self.learning_algorithm
    
    def evaluate(self, num_samples : int = 1):
        """
        Execute num_samples runs on the environment until completion and return
        the average cumulative reward.
        :param int num_samples: The number of sample runs executed in the environment
        to get the average reward. Defaults to 1.
        """
        # same thing to sample n cumulative rewards to sum and divide by n, 
        # and to sample a single cumulative reward for all runs and divide by n
        cumulative_reward = 0.0

        for _ in range(num_samples):

            s, _ = self.env_eval.reset() #reset the environment
            terminated = truncated = False
            
            # Evaluation loop:
            while not (terminated or truncated):

                # get optimal action by agent
                a = torch.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    self.learning_algorithm(np.expand_dims(s, 0)).cpu().detach()
                ).numpy()

                # update env accordingly
                s, r, terminated, truncated, _ = self.env_eval.step(a)
                cumulative_reward += r
            
        return cumulative_reward / num_samples



# class GymOffPolicyBaseTrainer:
    

#     def __init__(self, env, learning_algorithm):
#         self.env = env
#         self.learn_algo = learning_algorithm

#     def train(
#             self,
#             episodes : int,
#             exploration_episodes : int = None,
#             show_progress_every_n_episodes : int = None,
#             render_training : bool = False,
#             save_training_result : bool = True,
#             ):

#         # setup agent, which sets up both: 
#         # - the environment, and 
#         # - the learning algorithm
        
#         prior_reward = 0
#         reward_over_time = []

#         # Single episode loop:
#         for episode in tqdm.tqdm(range(episodes)):
#             # training:
#             # 0) reset the environment (env), and setup appropriate values: 
#             # - state of env
#             # - rewards over episode (0)
#             # - done or not? which is False
#             # - array storing pairs of state, action and next state 
#             # - epsilon for epsilon-greedy, which decreases by initial_epsilon / episodes every episode

#             s, _ = self.env.reset()
#             d = False
#             episode_reward = 0

#             # Training loop:
#             while not d:
                
#                 a = self.learn_algo(s)

#                 # -) update the environment accordingly given the action, taking: 
#                 # new state, new reward, done?, info
#                 n_s, r, terminated, truncated, _ = self.env.step(a)
#                 episode_reward += r

#                 if terminated or truncated:
#                     d = True

#                 # -) update info to adjust at the end of the step
#                 s = n_s
            
#             # Once episode is over:
#             # Update learning algorithm
            
#             self.learn_algo.update(experiences)

#             # Then adjust values accordingly
#             reward_over_time.append(episode_reward)
            
#             if episode % show_progress_every_n_episodes == 0:
#                 self.evaluate(env_agent_merged_object, agent)

#         # End of training things
#         self.env.close() # close the training env
#         if save_training_result:
#             self.learn_algo.save(task_name="SAC_BipedalWalker")
        
#         _, ax = plt.subplots()
#         ax.plot(reward_over_time, linewidth=2.0)
#         plt.show()

#         return self.learn_algo

#     #+++++++++++++++++++++++++++++++++++++++
#     #Evaluating

#     # First, define a function allowing to save gifs
#     # Great code from here[https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553]!
#     def save_frames_as_gif(frames, path='./training_gifs/', filename='animation.gif'):

#         #Mess with this to change frame size
#         plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

#         patch = plt.imshow(frames[0])
#         plt.axis('off')

#         def animate(i):
#             patch.set_data(frames[i])

#         anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#         anim.save(path + filename, writer='imagemagick', fps=60)

#     # Runs a full cycle of the environment given the agent or a path. 
#     # If a path is given, the agent object will load the path.
#     def evaluate(
#             env_agent_merged_object,
#             trained_agent,
#             path : str = None
#             ):
        
#         # load the path if it is given and not None 
#         if path != None:
#             try: 
#                 trained_agent.load(path)
#                 print("\n path has been loaded! \n")
#             except:
#                 raise Exception("Something went wrong when loading the path into the agent...")

#         # Evaluation loop:
#         env_eval = env_agent_merged_object(r_m="human")
#         s, _ = env_eval.reset() #reset the environment
#         terminated = truncated = False

#         while not (terminated or truncated):

#             # get optimal action by agent
#             a = trained_agent.get_optimal_action(s)

#             # update env accordingly
#             s, _, terminated, truncated, _ = env_eval.step(a)

#         env_eval.close()


#     def evaluate_while_returning_gif(
#             env_agent_merged_object,
#             trained_agent,
#             path : str = None,
#             gif_name : str = 'animation.gif'
#             ):
        
#         # load the path if it is given and not None 
#         if path != None:
#             try: 
#                 trained_agent.load(path)
#                 print("\n path has been loaded! \n")
#             except:
#                 raise Exception("Something went wrong when loading the path into the agent...")
        
#         # stores frames to be stored as gif
#         frames = []

#         # Evaluation loop:
#         env_eval = env_agent_merged_object(r_m="rgb_array")
#         s, _ = env_eval.reset() #reset the environment
#         terminated = truncated = False

#         while not (terminated or truncated):

#             # get optimal action by agent
#             a = trained_agent.get_optimal_action(s)

#             # update env accordingly
#             s, _, terminated, truncated, _ = env_eval.step(a)
            
#             #Render to frames buffer
#             frames.append(env_eval.env.render())

#         env_eval.close()
        
#         save_frames_as_gif(frames=frames, filename=gif_name)