"""
An abstract environment-agnostic base trainer class to be inherited from.
"""

from abc import ABC, abstractmethod
import logging
from timeit import default_timer as timer
import traceback
from typing import List

import matplotlib.pyplot as plt

from models.trainers.utils.buffer import Buffer, NdArrayBuffer
from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

class OffPolicyBaseTrainer(ABC):

    BUFFER_IMPLEMENTATION = NdArrayBuffer

    @abstractmethod
    def __init__(self, env):
        """
        Initializes an abstract off-policy base trainer class which takes
        an environment to use for algorithm training.

        :param env: The environment used for training.
        """
        self.env = env
    
    @abstractmethod
    def generate_experience(
        self,
        buffer : Buffer,
        exploration_function,
        learning_algorithm : PolicyLearningAlgorithm
        )-> int:
        """
        An abstract method which generates experience at the smallest scale,
        i.e. (number of parallel environments) x (smallest number of steps),
        and adds it to a passed buffer.
        Also returns the number of newly generated experiences.

        :param Buffer buffer: The buffer to which we add generated experience.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param PolicyLearningAlgorithm learning_algorithm: The algorithm that gives us 
        policy to collect experience with.
        :returns int: The number of newly generated experiences.
        """
        pass

    def generate_batch_of_experiences(
            self,
            buffer : Buffer,
            buffer_size : int,
            exploration_function,
            learning_algorithm : PolicyLearningAlgorithm
        ):
        """
        An abstract function which takes a buffer and fills it with "buffer_size" 
        random experiences sampled from running the policy in the environment.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.

        :param Buffer buffer: The buffer to which we add newly generated experiences.
        :param int buffer_size: The size of the buffer to be returned.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experience.
        :param OffPolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
        evaluating actions given states per batches.
        """
        # While we have not generated enough experience, keep generating
        num_generated_experiences = 0

        while num_generated_experiences < buffer_size:
            num_generated_experiences += self.generate_experience(buffer, 
                                                                  exploration_function, 
                                                                  learning_algorithm)
    
    @abstractmethod
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
            save_after_training : bool,
            task_name : str
            )-> PolicyLearningAlgorithm:
        """
        An abstract function which trains the given algorithm with given 
        parameter specifications.

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
        :param bool save_after_training: Whether to save the resulting algorithm using
        PolicyLearningAlgorithm.save().
        :param str task_name: The name of this task.
        :return PolicyLearningAlgorithm: The trained algorithm.
        """
        start_time = timer()
        
        try:
            experiences : Buffer = OffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=max_buffer_size,
                                                                                 obs_shape=self.obs_shape,
                                                                                 act_shape=self.act_shape)
            cumulative_rewards : List[float] = []

            # first populate the buffer with num_initial_experiences experiences
            print(f"Generating {num_initial_experiences} initial experiences...")
            self.generate_batch_of_experiences(
                buffer=experiences,
                buffer_size=num_initial_experiences,
                exploration_function=initial_exploration_function,
                learning_algorithm=learning_algorithm
            )
            print("Generation successful!")
                            
            # then go into the training loop
            for i in range(num_training_epochs):
                
                self.generate_batch_of_experiences(
                    buffer=experiences,
                    buffer_size=new_experience_per_epoch,
                    exploration_function=training_exploration_function,
                    learning_algorithm=learning_algorithm
                )
                
                learning_algorithm.update(experiences)

                # evaluate sometimes
                if (i + 1) % (evaluate_every_N_epochs) == 0:
                    cumulative_reward = self.evaluate(learning_algorithm, evaluate_N_samples)
                    cumulative_rewards.append(cumulative_reward)
                    print(f"Training loop {i+1}/{num_training_epochs} successfully ended: reward={cumulative_reward}.\n")
                
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
            print("Successfully closed envs!")

            print(f"Execution time: {end_time - start_time} sec.") #float fractional seconds?

            # Show the training graph
            try:
                plt.clf()
                plt.plot(range(0, len(cumulative_rewards)*evaluate_every_N_epochs, evaluate_every_N_epochs), cumulative_rewards)
                plt.savefig(f"{task_name}_cumulative_reward_fig.png")
                plt.show()
                learning_algorithm.show_loss_history(task_name=task_name, save_figure=True, save_dir=None)
            except Exception:
                logging.error(traceback.format_exc())
    
            return learning_algorithm

    @abstractmethod
    def evaluate(
        self,
        learning_algorithm : PolicyLearningAlgorithm, 
        num_samples : int = 5
        )-> float:
        """
        An abstract function which evaluates the learning_algorithm some 
        way or the other on the environment. Returns a cumulative reward
        obtained from the evaluation.

        :param PolicyLearningAlgorithm learning_algorithm: The algorithm to be evaluated.
        :param int num_samples: The number of "samples" to consider when doing the 
        evaluation to get an average estimated result, defaults to 5
        :return float: The cumulative reward as result of evaluation.
        """
        pass