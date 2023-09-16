"""
Represents a base trainer which can be adapted to any gym environment
by making a trainer which inherits from it.

~Classes~
 BaseTrainer: a handler of interaction between the learning algorithm and the gym environment
              to collect experience and update the algorithm
"""

import logging
from timeit import default_timer as timer
import traceback
from typing import List

import gymnasium
import matplotlib.pyplot as plt
import numpy as np

from models.trainers.utils.buffer import Buffer, NdArrayBuffer
from models.trainers.utils.experience import Experience

class GymOnPolicyBaseTrainer:
    pass

class GymOffPolicyBaseTrainer:

    BUFFER_IMPLEMENTATION = NdArrayBuffer

    def __init__(
            self, 
            env
        ):
        """
        Creates an on policy base trainer with a given gym environment and a 
        learning algorithm which holds a policy.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param env: The gymnasium environment used.
        """
        self.env = env
        
        self.env_eval_render = gymnasium.make(self.env.spec.id, render_mode="human")
        self.env_eval_no_render = gymnasium.make(self.env.spec.id)

        self.reset_trainer()

        # prints specs about the environment
        observation_size = 1 if (env.observation_space.shape == ()) else env.observation_space.shape[0]
        action_size      = 1 if (env.action_space.shape == ())      else env.action_space.shape[0]
        print(f"The environment has observation size: {observation_size} & action size: {action_size}.")
    
    def reset_trainer(self):
        """
        Reset the environment and internal variables.
        """
        self.last_action : np.ndarray = None
        self.last_observation : np.ndarray = None

    def generate_experience(
            self,
            exploration_function, 
            learning_algorithm
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
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
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
        # if self.last_action is None (beginning of training), we reset env and take a step first
        if self.last_action == None:
            self.last_observation, _ = self.env.reset()
            self.last_action = exploration_function(
                np.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    learning_algorithm(np.expand_dims(self.last_observation, 0)), 
                    axis=0
                ), 
                self.env
            )

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
                np.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    learning_algorithm(np.expand_dims(self.last_observation, 0)), 
                    axis=0
                ),
                self.env
            )
            
            # Store info of action picked to generate new experience in the next loop
            self.last_action = best_action
        else:
            # reset the state of this trainer (not the environment!)
            self.reset_trainer()

        return new_experience    

    def generate_batch_of_experiences(
            self, 
            buffer_size : int,
            exploration_function,
            learning_algorithm
        ) -> Buffer:
        """
        Generates and returns a buffer containing "buffer_size" random experiences 
        sampled from running the policy in the environment.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.

        :param int buffer_size: The size of the buffer to be returned.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experience.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :returns Buffer buffer: The buffer containing buffer_size experiences.
        """

        # Create an empty buffer
        buffer : Buffer = GymOffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=buffer_size, 
                                                                        obs_shape=self.env.observation_space.shape, 
                                                                        act_shape=self.env.action_space.shape)

        # While we have not enough experience in buffer, generate experience
        while buffer.size() < buffer_size:
            new_experience = self.generate_experience(exploration_function, learning_algorithm)
            
            buffer.append_experience(new_experience.obs, new_experience.action, 
                                 new_experience.reward, new_experience.done, 
                                 new_experience.next_obs)

        return buffer
    
    def train(
            self,
            learning_algorithm,
            num_training_steps : int,
            num_new_experience : int,
            max_buffer_size : int,
            num_initial_experiences : int,
            evaluate_every_N_steps : int,
            initial_exploration_function,
            training_exploration_function,
            save_after_training : bool,
            task_name : str,
            render_evaluation : bool = True
        ):
        """
        Trains the learning algorithm in the environment with given specifications, returning 
        the trained algorithm.

        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :param int num_training_steps: The number of steps we proceed to train the algorithm.
        :param int num_new_experience: The number of new experience we collect minimum before
        every update for our algorithm.
        :param int max_buffer_size: The maximum number of experience to be kept in the buffer.
        :param int num_initial_experiences: The number of experiences to be collected in the 
        buffer first before doing any learning.
        :param int evaluate_every_N_steps: We evaluate the algorithm at this interval.
        :param initial_exploration_function: A function which controls how exploration is 
        handled at the very beginning, when exploration is generated.
        :param training_exploration_function: A function which controls how exploration is handled 
        during training of the algorithm.
        :param bool save_after_training: Whether to save the policy after training.
        :param str task_name: The name of the task to log when saving after training.
        :param bool render_evaluation: Whether to render the environment when evaluating.
        """

        start_time = timer()
        
        try:
            experiences : Buffer = GymOffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=max_buffer_size,
                                                                                 obs_shape=self.env.observation_space.shape, 
                                                                                 act_shape=self.env.action_space.shape)
            cumulative_rewards : List[float] = []
            loss_history : List[float] = []

            # first populate the buffer with num_initial_experiences experiences
            print(f"Generating {num_initial_experiences} initial experiences...")
            init_exp = self.generate_batch_of_experiences(
                buffer_size=num_initial_experiences,
                exploration_function=initial_exploration_function,
                learning_algorithm=learning_algorithm
            )
            print("Generation successful!")
            # TODO TOREMOVE

            # print("Action generated at the very beginning!")
            # obs = torch.unsqueeze(torch.from_numpy(init_exp[0].obs), dim=0).cuda()
            # act = torch.tensor([[-0.1313,  0.0119, -0.3108,  0.0566],]).cuda()
            # print("obs: ", obs, "obs.shape: ", obs.shape, "act: ", act, "act.shape: ", act.shape)
            # pred = torch.squeeze(self.learning_algorithm.qnet1(obs=obs, actions=act), dim=1)
            # print("pred: ", pred)
            
            # END TOREMOVE
            
            experiences.extend_buffer(init_exp)
                
            # then go into the training loop
            for i in range(num_training_steps):
                
                new_exp = self.generate_batch_of_experiences(
                    buffer_size=num_new_experience,
                    exploration_function=training_exploration_function,
                    learning_algorithm=learning_algorithm
                )
                
                experiences.extend_buffer(new_exp)
                loss = learning_algorithm.update(experiences)
                loss_history.append(loss)

                # evaluate sometimes
                if (i + 1) % (evaluate_every_N_steps) == 0:
                    cumulative_reward = self.evaluate(learning_algorithm, 1, render=render_evaluation)
                    cumulative_rewards.append(cumulative_reward)
                    print(f"Training loop {i+1}/{num_training_steps} successfully ended: reward={cumulative_reward}.\n")
                
            if save_after_training: learning_algorithm.save(task_name)
            print("Training ended successfully!")

        except KeyboardInterrupt:
            print("\nTraining interrupted, continue to next cell to save to save the model.")
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            end_time = timer()

            print("Closing envs...")
            self.env.close()
            self.env_eval_render.close()
            self.env_eval_no_render.close()
            print("Successfully closed envs!")

            print(f"Execution time: {end_time - start_time} sec.") #float fractional seconds?

            # Show the training graph
            try:
                plt.clf()
                plt.plot(range(0, len(cumulative_rewards)*evaluate_every_N_steps, evaluate_every_N_steps), cumulative_rewards)
                plt.savefig(f"{task_name}_cumulative_reward_fig.png")
                plt.show()

                plt.clf()
                plt.plot(range(0, len(loss_history)), loss_history)
                plt.savefig(f"{task_name}_loss_history_fig.png")
                plt.show()
            except ValueError:
                print("\nPlot failed on interrupted training.")
                
            return learning_algorithm
    
    def evaluate(self, learning_algorithm, num_samples : int = 5, render : bool = True):
        """
        Execute num_samples runs on the environment until completion and return
        the average cumulative reward.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :param int num_samples: The number of sample runs executed in the environment
        to get the average reward. Defaults to 1.
        :param bool render: Whether to render the enviroment during evaluation.
        Defaults to True.
        """
        env = self.env_eval_render if render else self.env_eval_no_render

        # same thing to sample n cumulative rewards to sum and divide by n, 
        # and to sample a single cumulative reward for all runs and divide by n
        cumulative_reward = 0.0

        for _ in range(num_samples):

            s, _ = env.reset() #reset the environment
            terminated = truncated = False
            
            # Evaluation loop:
            while not (terminated or truncated):

                # get optimal action by agent
                a = np.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    learning_algorithm(np.expand_dims(s, 0)), axis=0
                )

                # update env accordingly
                s, r, terminated, truncated, _ = env.step(a)
                cumulative_reward += r
            
        return cumulative_reward / num_samples