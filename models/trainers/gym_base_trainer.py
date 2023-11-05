"""
Represents a base trainer which can be adapted to any gym environment
by making a trainer which inherits from it.

~Classes~
 BaseTrainer: a handler of interaction between the learning algorithm and the gym environment
              to collect experience and update the algorithm
"""

import gymnasium
import numpy as np

from models.trainers.base_trainer import OffPolicyBaseTrainer
from models.trainers.utils.buffer import Buffer
from models.trainers.utils.experience import Experience
from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

class GymOnPolicyBaseTrainer:
    pass

class GymOffPolicyBaseTrainer(OffPolicyBaseTrainer):

    def __init__(
            self,
            env : gymnasium.Env
        ):
        """
        Creates an on policy base trainer with a given gym environment and a 
        learning algorithm which holds a policy.
        Enables the generation of the experience for a single time step and return it
        to be used for off-policy learning.

        :param gymnasium.Env env: The gymnasium environment used.
        """
        if not isinstance(env, gymnasium.Env):
            raise Exception("The environment passed to GymOffPolicyBaseTrainer was not a gymnasium " + 
                            "environment - please make sure that a gymnasium environment is passed!")

        super().__init__(env=env)
        self.reset_trainer()
        
        self.obs_shape=self.env.observation_space.shape
        self.act_shape=self.env.action_space.shape
        self.env_name = env.unwrapped.spec.id
    
        # prints specs about the environment
        observation_size = 1 if (self.obs_shape == ()) else self.obs_shape[0]
        action_size      = 1 if (self.act_shape == ()) else self.act_shape[0]
        print(f"The environment has observation size: {observation_size} & action size: {action_size}.")
    
    def reset_trainer(self):
        """
        Reset the environment and internal variables.
        """
        self.last_action : np.ndarray = None
        self.last_observation : np.ndarray = None

    # def generate_experience(
    #         self,
    #         buffer : Buffer,
    #         exploration_function, 
    #         learning_algorithm : PolicyLearningAlgorithm
    #         ) -> Experience:
    #     """
    #     Executes a single step in the environment to generate experiences,
    #     and adds them to a passed buffer.
    #     The experience involves some exploration (vs. exploitation) which behavior 
    #     is determined by exploration_function.
    #     Also returns the number of newly generated experiences.
        
    #     :param Buffer buffer: The buffer to which we add generated experience.
    #     :param exploration_function: A function which controls how exploration is handled 
    #     during collection of experiences.
    #     :param PolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
    #     evaluating actions given states per batches.
    #     :returns int: The number of newly generated experiences.
    #     """
        
    #     # if self.last_action is None (beginning of training), we reset env and take a step first
    #     if type(self.last_action) == type(None):
    #         self.last_observation, _ = self.env.reset()
    #         self.last_action = exploration_function(
    #             np.squeeze( #adjust, as output from learning_algo always has a batch dimension
    #                 learning_algorithm(np.expand_dims(self.last_observation.copy(), 0)), 
    #                 axis=0
    #             ), 
    #             self.env
    #         )

    #     new_observation, reward, terminated, truncated, _ = self.env.step(self.last_action.copy())
            
    #     # generate the new experience based on the last stored info and the new info,
    #     # adding it to the passed buffer
    #     buffer.append_experience(obs=self.last_observation.copy(), act=self.last_action.copy(), 
    #                              rew=reward, don=terminated, next_obs=new_observation.copy())
    #     # update last observation
    #     self.last_observation = new_observation

    #     # if the environment hasn't ended
    #     if not (terminated or truncated):
    #         # Generate actions for agents while applying the exploration function to 
    #         # promote exploration of the world
            
    #         best_action = exploration_function(
    #             np.squeeze( #adjust, as output from learning_algo always has a batch dimension
    #                 learning_algorithm(np.expand_dims(self.last_observation.copy(), 0)), 
    #                 axis=0
    #             ),
    #             self.env
    #         )
            
    #         # Store info of action picked to generate new experience in the next loop
    #         self.last_action = best_action
    #     else:
    #         # reset the state of this trainer (not the environment!)
    #         self.reset_trainer()

    #     return 1 #for the above code, we only produce one experience per step


    def generate_experience(
            self,
            buffer : Buffer,
            exploration_function, 
            learning_algorithm : PolicyLearningAlgorithm
            ) -> Experience:
        """
        Executes a single step in the environment to generate experiences,
        and adds them to a passed buffer.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.
        Also returns the number of newly generated experiences.
        
        :param Buffer buffer: The buffer to which we add generated experience.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param PolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
        evaluating actions given states per batches.
        :returns int: The number of newly generated experiences.
        """

        # if self.last_action is None (beginning of training), we reset env and take a step first
        if type(self.last_observation) == type(None):
            self.last_observation, _ = self.env.reset()

        action = exploration_function(
            np.squeeze( #adjust, as output from learning_algo always has a batch dimension
                learning_algorithm(np.expand_dims(self.last_observation.copy(), 0)), 
                axis=0
                ), 
                self.env
            )
        
        new_observation, reward, terminated, truncated, _ = self.env.step(action.copy())
            
        # generate the new experience based on the last stored info and the new info,
        # adding it to the passed buffer
        buffer.append_experience(obs=self.last_observation.copy(), act=action.copy(), 
                                 rew=reward, don=terminated, next_obs=new_observation.copy())
        # update last observation
        self.last_observation = new_observation

        # if the environment has ended
        if terminated or truncated:
            # reset the state of this trainer (not the environment!)
            self.reset_trainer()

        return 1 #for the above code, we only produce one experience per step
    
    def train(
            self,
            learning_algorithm : PolicyLearningAlgorithm,
            num_training_epochs : int,
            new_experience_per_epoch : int,
            max_buffer_size : int,
            num_initial_experiences : int,
            evaluate_every_N_epochs : int,
            evaluate_N_samples : int,
            initial_exploration_function,
            training_exploration_function,
            training_exploration_function_name : str,
            save_after_training : bool,
            task_name : str,
            training_id : int,
            render_evaluation : bool = True
            )-> PolicyLearningAlgorithm:
        """
        Trains the learning algorithm in the environment with given specifications, returning 
        the trained algorithm.

        :param PolicyLearningAlgorithm learning_algorithm: The algorithm to train.
        :param int num_training_epochs: How many training epochs are executed.
        :param int new_experience_per_epoch: The number of minimum experience to 
        produce per epoch. *actual number of newly produced experience might 
        differ since the smallest increment of the number of experience matches that
        produced by generate_experience. 
        :param int max_buffer_size: The maximum number of experience to keep in buffer.
        :param int num_initial_experiences: The number of experiences added to buffer 
        before training starts.
        :param int evaluate_every_N_epochs: How frequently we evaluate the algorithm
        using self.evaluate.
        :param int evaluate_N_samples: How many samples of cumulative reward trajectory you 
        average for evaluate.
        :param initial_exploration_function: The exploration function used when producing
        the initial experiences.
        :param training_exploration_function: The exploration function used when producing
        experiences while training the algorithm.
        :param str training_exploration_function_name: The training exploration function's name.
        :param bool save_after_training: Whether to save the resulting algorithm using
        PolicyLearningAlgorithm.save().
        :param str task_name: The name of this task.
        :param int training_id: The training id that specifies the training process.
        :param bool render_evaluation: Whether to render the environment when evaluating.

        :return PolicyLearningAlgorithm: The trained algorithm.
        """
        self.render_evaluation = render_evaluation

        return super().train(
            learning_algorithm=learning_algorithm, 
            num_training_epochs=num_training_epochs, 
            new_experience_per_epoch=new_experience_per_epoch,
            max_buffer_size=max_buffer_size,
            num_initial_experiences=num_initial_experiences,
            evaluate_every_N_epochs=evaluate_every_N_epochs,
            evaluate_N_samples=evaluate_N_samples,
            initial_exploration_function=initial_exploration_function,
            training_exploration_function=training_exploration_function,
            training_exploration_function_name=training_exploration_function_name,
            save_after_training=save_after_training, 
            task_name=task_name,
            training_id=training_id
            )
    
    def evaluate(self, 
                 learning_algorithm : PolicyLearningAlgorithm, 
                 num_samples : int = 5) -> float:
        """
        Execute num_samples runs on the environment until completion and return
        the average cumulative reward.
        :param OffPolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
        evaluating actions given states per batches.
        :param int num_samples: The number of sample runs executed in the environment
        to get the average reward. Defaults to 1.
        :returns float: Returns the average cumulative reward.
        """
        def evaluate_on_environment(env):
            cum_rew = 0.0
            s, _ = env.reset() #reset the environment
            terminated, truncated = False, False
            
            # Evaluation loop:
            while not (terminated or truncated):

                # get optimal action by agent
                a = np.squeeze( #adjust, as output from learning_algo always has a batch dimension
                    learning_algorithm(np.expand_dims(s, 0)), axis=0
                )

                # update env accordingly
                s, r, terminated, truncated, _ = env.step(a)
                cum_rew += r
            
            return cum_rew
        
        # same thing to sample n cumulative rewards to sum and divide by n, 
        # and to sample a single cumulative reward for all runs and divide by n
        cumulative_reward = 0.0
        
        # visualize and render the first evaluation loop
        env_eval = gymnasium.make(self.env.spec.id, 
                                  render_mode="human" if self.render_evaluation else None)
        cumulative_reward += evaluate_on_environment(env_eval)
        env_eval.close()
        
        # remaining evaluation without rendering
        env_eval = gymnasium.make(self.env.spec.id)

        for _ in range(num_samples - 1):
            cumulative_reward += evaluate_on_environment(env_eval)
        
        env_eval.close()

        return cumulative_reward / num_samples