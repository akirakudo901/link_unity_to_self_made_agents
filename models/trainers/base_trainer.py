"""
An abstract environment-agnostic base trainer class to be inherited from.
"""

from abc import ABC, abstractmethod
import logging
import os
import shutil
from timeit import default_timer as timer
import traceback
from typing import List

import matplotlib.pyplot as plt
import yaml

from models.trainers.utils.buffer import Buffer, NdArrayBuffer
from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

class OffPolicyBaseTrainer(ABC):

    BUFFER_IMPLEMENTATION = NdArrayBuffer
    PROGRESS_SAVING_DIR = os.path.join("trained_algorithms", "_in_progress")

    @abstractmethod
    def __init__(self, env):
        """
        Initializes an abstract off-policy base trainer class which takes
        an environment to use for algorithm training.

        :param env: The environment used for training.
        """
        self.env = env
        self.env_name = ""
    
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
            training_exploration_function_name : str,
            save_after_training : bool,
            task_name : str,
            training_id : int
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
        :param str training_exploration_function_name: The training exploration function's name.
        :param bool save_after_training: Whether to save the resulting algorithm using
        PolicyLearningAlgorithm.save().
        :param str task_name: The name of this task.
        :param int training_id: The training id that specifies the training process.
        :return PolicyLearningAlgorithm: The trained algorithm.
        """
        # first try to resume a previous training if there is one
        if os.path.exists(os.path.join(OffPolicyBaseTrainer.PROGRESS_SAVING_DIR, 
                                       f"{task_name}_{training_id}_folder")):
            try:
                algo = self.resume_training(learning_algorithm=learning_algorithm,
                                            training_exploration_function=training_exploration_function,
                                            task_name=task_name,
                                            training_id=training_id,
                                            training_exploration_function_name=training_exploration_function_name)
                return algo
            except Exception:
                logging.error(traceback.format_exc())
                print("Loading a previous progress was unsuccessful...")

        # otherwise train from scratch
        start_time = timer()
        
        try:
            experiences : Buffer = OffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=max_buffer_size,
                                                                              obs_shape=self.obs_shape,
                                                                              act_shape=self.act_shape)
            
            # first populate the buffer with num_initial_experiences experiences
            print(f"Generating {num_initial_experiences} initial experiences...")
            self.generate_batch_of_experiences(
                buffer=experiences,
                buffer_size=num_initial_experiences,
                exploration_function=initial_exploration_function,
                learning_algorithm=learning_algorithm
            )
            print("Generation successful!")

            end_time = timer()
            learning_algorithm = self._helper_train(learning_algorithm=learning_algorithm,
                                                    num_training_epochs=num_training_epochs,
                                                    training_epochs_so_far=0,
                                                    new_experience_per_epoch=new_experience_per_epoch,
                                                    evaluate_every_N_epochs=evaluate_every_N_epochs,
                                                    evaluate_N_samples=evaluate_N_samples,
                                                    training_exploration_function=training_exploration_function,
                                                    training_exploration_function_name=training_exploration_function_name,
                                                    save_after_training=save_after_training,
                                                    task_name=task_name,
                                                    training_id=training_id,
                                                    experiences=experiences,
                                                    time_so_far=end_time - start_time,
                                                    cumulative_rewards=[])
            return learning_algorithm
        
        except Exception:
            logging.error(traceback.format_exc())
        
    # @abstractmethod
    # def train(
    #         self,
    #         learning_algorithm : PolicyLearningAlgorithm,
    #         num_training_epochs : int,
    #         new_experience_per_epoch : int,
    #         max_buffer_size : int,
    #         num_initial_experiences : int,
    #         evaluate_every_N_epochs : int,
    #         evaluate_N_samples : int,
    #         initial_exploration_function,
    #         training_exploration_function,
    #         save_after_training : bool,
    #         task_name : str
    #         )-> PolicyLearningAlgorithm:
    #     """
    #     An abstract function which trains the given algorithm with given 
    #     parameter specifications.

    #     :param PolicyLearningAlgorithm learning_algorithm: The algorithm to train.
    #     :param int num_training_epochs: How many training epochs are executed.
    #     :param int new_experience_per_epoch: The number of minimum experience to 
    #     produce per epoch. *actual number of newly produced experience might 
    #     differ since the smallest increment of the number of experience matches that
    #     produced by generate_experience. 
    #     :param int max_buffer_size: The maximum number of experience to keep in buffer.
    #     :param int num_initial_experiences: The number of experiences added to buffer 
    #     before training starts.
    #     :param int evaluate_every_N_epochs: How frequently we evaluate the algorithm
    #     using self.evaluate.
    #     :param int evaluate_N_samples: How many samples of cumulative reward trajectory you 
    #     average for evaluate.
    #     :param initial_exploration_function: The exploration function used when producing
    #     the initial experiences.
    #     :param training_exploration_function: The exploration function used when producing
    #     experiences while training the algorithm.
    #     :param bool save_after_training: Whether to save the resulting algorithm using
    #     PolicyLearningAlgorithm.save().
    #     :param str task_name: The name of this task.
    #     :return PolicyLearningAlgorithm: The trained algorithm.
    #     """
    #     start_time = timer()
        
    #     try:
    #         experiences : Buffer = OffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=max_buffer_size,
    #                                                                           obs_shape=self.obs_shape,
    #                                                                           act_shape=self.act_shape)
    #         cumulative_rewards : List[float] = []

    #         # first populate the buffer with num_initial_experiences experiences
    #         print(f"Generating {num_initial_experiences} initial experiences...")
    #         self.generate_batch_of_experiences(
    #             buffer=experiences,
    #             buffer_size=num_initial_experiences,
    #             exploration_function=initial_exploration_function,
    #             learning_algorithm=learning_algorithm
    #         )
    #         print("Generation successful!")
                            
    #         # then go into the training loop
    #         for i in range(num_training_epochs):
                
    #             self.generate_batch_of_experiences(
    #                 buffer=experiences,
    #                 buffer_size=new_experience_per_epoch,
    #                 exploration_function=training_exploration_function,
    #                 learning_algorithm=learning_algorithm
    #             )
                
    #             learning_algorithm.update(experiences)

    #             # evaluate sometimes
    #             if (i + 1) % (evaluate_every_N_epochs) == 0:
    #                 cumulative_reward = self.evaluate(learning_algorithm, evaluate_N_samples)
    #                 cumulative_rewards.append(cumulative_reward)
    #                 print(f"Training loop {i+1}/{num_training_epochs} successfully ended: reward={cumulative_reward}.\n")
                
    #         if save_after_training: learning_algorithm.save(task_name)
    #         print("Training ended successfully!")

    #     except KeyboardInterrupt:
    #         print("\nTraining interrupted, continue to next cell to save to save the model.")
    #     except Exception:
    #         logging.error(traceback.format_exc())
    #     finally:
    #         end_time = timer()

    #         print("Closing envs...")
    #         self.env.close()
    #         print("Successfully closed envs!")

    #         print(f"Execution time: {end_time - start_time} sec.") #float fractional seconds?

    #         # Show the training graph
    #         try:
    #             plt.clf()
    #             plt.title(f"{task_name} Cumulative reward")
    #             plt.xlabel("Epochs")
    #             plt.ylabel("Cumulative Reward")
    #             plt.plot(range(0, len(cumulative_rewards)*evaluate_every_N_epochs, evaluate_every_N_epochs), cumulative_rewards)
    #             plt.savefig(f"{task_name}_cumulative_reward_fig.png")
    #             plt.show()
    #             learning_algorithm.show_loss_history(task_name=task_name, save_figure=True, save_dir=None)
    #         except Exception:
    #             logging.error(traceback.format_exc())
    
    #         return learning_algorithm
    
    def _helper_train(self,
                      learning_algorithm : PolicyLearningAlgorithm,
                      num_training_epochs : int,
                      training_epochs_so_far : int,
                      new_experience_per_epoch : int,
                      evaluate_every_N_epochs : int,
                      evaluate_N_samples : int,
                      training_exploration_function,
                      training_exploration_function_name : str,
                      save_after_training : bool,
                      task_name : str,
                      training_id : int,
                      experiences : Buffer,
                      time_so_far : float,
                      cumulative_rewards : List[float]
                      )-> PolicyLearningAlgorithm:
        """
        Helper to abstract code from both train() and resume_training().
        """
        start_time = timer()
        
        try:

            # the training loop
            for i in range(training_epochs_so_far, num_training_epochs):
                
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

                    intermediate_time = timer()

                    if save_after_training:
                        # also save when evaluating
                        self._save_training_progress(
                            task_name=task_name,
                            training_id=training_id,
                            learning_algorithm=learning_algorithm,
                            buffer=experiences,
                            num_training_epochs=num_training_epochs,
                            new_experience_per_epoch=new_experience_per_epoch,
                            evaluate_every_N_epochs=evaluate_every_N_epochs,
                            evaluate_N_samples=evaluate_N_samples,
                            training_exploration_function_name=training_exploration_function_name,
                            save_after_training=save_after_training,
                            training_epochs_so_far=i,
                            time_so_far=time_so_far + (intermediate_time - start_time),
                            cumulative_rewards=cumulative_rewards,
                            environment_name=self.env_name
                        )
                
            if save_after_training: learning_algorithm.save(task_name)
            print("Training ended successfully!")

        except KeyboardInterrupt:
            print("\nTraining interrupted...")
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            end_time = timer()

            print("Closing envs...")
            self.env.close()
            print("Successfully closed envs!")

            print(f"Execution time for this session: {end_time - start_time} sec.") #float fractional seconds?
            print(f"Execution time for the entire training so far: {end_time - start_time + time_so_far} sec.")

            # Show the training graph
            try:
                new_dir_path = os.path.join("images", f"{task_name}_{training_id}_folder")
                if not os.path.exists(new_dir_path): os.mkdir(new_dir_path)
                plt.clf()
                plt.title(f"{task_name} ID={training_id} Cumulative reward")
                plt.xlabel("Epochs")
                plt.ylabel("Cumulative Reward")
                plt.plot(range(0, len(cumulative_rewards)*evaluate_every_N_epochs, evaluate_every_N_epochs), cumulative_rewards)
                plt.savefig(os.path.join(new_dir_path, f"{task_name}_{training_id}_cumulative_reward_fig.png"))
                plt.show()
                learning_algorithm.show_loss_history(task_name=task_name, save_figure=True, 
                                                     save_dir=new_dir_path)
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

    def resume_training(self,
                        learning_algorithm : PolicyLearningAlgorithm,
                        training_exploration_function,
                        task_name : str,
                        training_id : int,
                        training_exploration_function_name : str
                        )->PolicyLearningAlgorithm:
        """
        Resumes training that has been previously saved based on the given
        task name and training id; or fail if no corresponding progress exist.

        :param PolicyLearningAlgorithm learning_algorithm: A policy learning algorithm \
        corresponding in class to the one previously used - this function takes care of \
        reloading it.
        :param training_exploration_function: The function to be used for exploration.
        :param str task_name: The task name to be resumed.
        :param int training_id: The training id to be resumed.
        :param str training_exploration_function_name: The training exploration function's name.
        :return PolicyLearningAlgorithm: The algorithm that was trained.
        """
        learning_algorithm, buffer, param_dict = self._load_training_progress(
            task_name=task_name,
            training_id=training_id,
            learning_algorithm=learning_algorithm
            )
        print(f"The previous training exploration function name was: {param_dict['training_exploration_function_name']}.")
        print(f"The previous environment name was: {param_dict['environment_name']}.")

        learning_algorithm = self._helper_train(learning_algorithm=learning_algorithm,
                                                num_training_epochs=param_dict["num_training_epochs"],
                                                training_epochs_so_far=param_dict["training_epochs_so_far"],
                                                new_experience_per_epoch=param_dict["new_experience_per_epoch"],
                                                evaluate_every_N_epochs=param_dict["evaluate_every_N_epochs"],
                                                evaluate_N_samples=param_dict["evaluate_N_samples"],
                                                training_exploration_function=training_exploration_function,
                                                training_exploration_function_name=training_exploration_function_name,
                                                save_after_training=param_dict["save_after_training"],
                                                task_name=task_name,
                                                training_id=training_id,
                                                experiences=buffer,
                                                time_so_far=param_dict["time_so_far"],
                                                cumulative_rewards=param_dict["cumulative_rewards"])
        
    def _save_training_progress(
            self,
            task_name : str,
            training_id : int,
            learning_algorithm : PolicyLearningAlgorithm,
            buffer : Buffer,
            num_training_epochs : int,
            new_experience_per_epoch : int,
            evaluate_every_N_epochs : int,
            evaluate_N_samples : int,
            training_exploration_function_name : str,
            save_after_training : bool,
            training_epochs_so_far : int,
            time_so_far : float,
            cumulative_rewards : List[float],
            environment_name : str
    ):
        """
        Saves the current training progress into an intermediate
        form as combinations of YAML and pth files, with given
        task name and training id.

        :param str training_exploration_function_name: The name of the training \
        exploration function such that we can ensure the same function is (roughly)\
        used in the training.
        :param int training_epochs_so_far: The number of training epochs elapsed so far.
        :param float time_so_far: The time elapsed for training so far.
        :param List[float] cumulative_rewards: The cumulative rewards so far.
        :param str environment_name: The environment's name such that we can ensure at \
        least environments with the same name are run (unsure how to ensure that the \
        same environment is used for training).
        """
        new_dir_path = os.path.join(OffPolicyBaseTrainer.PROGRESS_SAVING_DIR,
                                    f"{task_name}_{training_id}_folder")
        if not os.path.exists(new_dir_path): os.mkdir(new_dir_path)
        # save learning state, comprising of:
        param_dict = {
            # 1/Elements given as parameters to train:
            "num_training_epochs" : num_training_epochs,
            "new_experience_per_epoch" : new_experience_per_epoch,
            "evaluate_every_N_epochs" : evaluate_every_N_epochs,
            "evaluate_N_samples" : evaluate_N_samples,
            "training_exploration_function_name" : training_exploration_function_name,
            "save_after_training" : save_after_training,
            "task_name" : task_name,
            # 2/Elements otherwise used to track progress
            "training_epochs_so_far" : training_epochs_so_far,
            "time_so_far" : time_so_far, #i.e. the time we have spent so far in training
            "cumulative_rewards" : cumulative_rewards,
            # 3/ The environment - just the name for now, to
            # ensure that we are restarting the training with the
            # correct environment later! 
            # ALL OF THE ABOVE WILL BE STORED AS YAML FOR NOW.
            "environment_name" : environment_name
        }
        with open(os.path.join(new_dir_path, f"{task_name}_{training_id}_trainer_parameters.yaml"),
                               'w') as file:
            yaml.dump(param_dict, file)

        # save policy learning algorithm
        learning_algorithm.save_training_progress(dir=new_dir_path, 
                                                  task_name=task_name, 
                                                  training_id=training_id)
        # save buffer
        buffer.save(save_dir=new_dir_path, 
                    file_name=f"{task_name}_{training_id}_buffer")
    
    def _load_training_progress(
            self,
            task_name : str,
            training_id : int,
            learning_algorithm : PolicyLearningAlgorithm
            ):
        """
        Loads the training progress identified by task name and 
        training id from the corresponding files.
        Raises an error if no such training progress exist.

        :param str task_name: The task name, e.g. Cartpole_DDQN
        :param int training_id: The training id distinguishing training data \
        among the same task name.
        :param PolicyLearningAlgorithm learning_algorithm: A learning algorithm that \
        is randomly initialized. Calling this function will initialize its content \
        appropriately for you.
        """
        new_dir_path = os.path.join(OffPolicyBaseTrainer.PROGRESS_SAVING_DIR,
                            f"{task_name}_{training_id}_folder")
        # load the learning state as a dictionary (to be directly returned?)
        with open(os.path.join(new_dir_path, f"{task_name}_{training_id}_trainer_parameters.yaml"),
                               'r') as file:
            #should be safe so far as we only load files created by this code;
            # if you do import codes from the outside, beware of YAML's 
            # building functionality that might be harmful.
            param_dict = yaml.load(file, Loader=yaml.Loader) 

        # load the policy learning algorithm
        learning_algorithm.load_training_progress(dir=new_dir_path, 
                                                  task_name=task_name, 
                                                  training_id=training_id)
        
        # load buffer
        buffer = NdArrayBuffer(max_size=1, obs_shape=(1, ), act_shape=(1, ))
        buffer.load(path=os.path.join(new_dir_path, f"{task_name}_{training_id}_buffer"))
        return (learning_algorithm, buffer, param_dict)
    
    def _delete_training_progress(
            self,
            task_name : str,
            training_id : int
            ):
        """
        Deletes the training progress identified by task name and 
        training id. Raises an error if no such training progress exist.

        :param str task_name: The task name.
        :param int training_id: The training id.
        """
        delete_dir_path = os.path.join(OffPolicyBaseTrainer.PROGRESS_SAVING_DIR,
                                       f"{task_name}_{training_id}_folder")
        if os.path.exists(delete_dir_path):
            shutil.rmtree(delete_dir_path)