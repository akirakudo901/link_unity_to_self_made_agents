"""
Represents a base trainer which can be adapted to any unity environment
by making a trainer which inherits from it.

~Classes~
 Experience:  a template for a single transition experience in the given world
 BaseTrainer: a handler of interaction between the learning algorithm and the unity environment
              to collect experience and update the algorithm
"""

import logging
import random
from timeit import default_timer as timer
import traceback
from typing import Dict, List, Tuple

import numpy as np

from mlagents_envs.environment import ActionTuple, BaseEnv, BehaviorName
import matplotlib.pyplot as plt

from models.trainers.utils.experience import Experience
from models.trainers.utils.buffer import NdArrayBuffer
from models.policy_learning_algorithms.policy_learning_algorithm import PolicyLearningAlgorithm

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]


class UnityOnPolicyBaseTrainer:

    BUFFER_IMPLEMENTATION = NdArrayBuffer

    def __init__(
            self, 
            env : BaseEnv, 
            behavior_name : BehaviorName
        ):
        """
        Creates an on policy base trainer with a given BaseEnv and BehaviorName specifying agent.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param BaseEnv env: The Unity environment used.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        """
        self.env = env
        self.behavior_name = behavior_name

        self.trajectory_by_agent : Dict[int, Trajectory] = {}
        self.cumulative_reward_by_agent : Dict[int, float] = {}
        self.last_action_by_agent : Dict[int, np.ndarray] = {}
        self.last_observation_by_agent : Dict[int, np.ndarray] = {}

    def generate_experience(
        self,
        exploration_function,
        learning_algorithm,
        behavior_name : BehaviorName=None
        ) -> Tuple[Buffer, List[Trajectory], List[float]]:
        """
        Executes a single step in the environment for every agent with behavior_name and
        according to policy by learning_algorithm, storing the last state & action, 
        trajectory so far, and cumulative rewards for each agent. 
        Returns the experience of agents in the last step, and trajectories & cumulative rewards
        for agents who reached a terminal step.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :param BehaviorName behavior_name: The behavior_name of the agent for which we 
        generate the experience.
        :returns Buffer experiences: The experiences of every agent that took action in
        the last step.
        :returns List[Trajectory] terminal_trajectories: The list of trajectories for agents
        that reached a terminal state.
        :returns List[float] terminal_cumulative_rewards: The list of rewards for agents that 
        reached a terminal state. 
        """
        if behavior_name is None: behavior_name = self.behavior_name
        experiences : Buffer = []
        terminal_trajectories : List[Trajectory] = []
        terminal_cumulative_rewards : List[float] = []
        """
        Generating one batch of trajectory comes in the following loop
        1) Get info from environment via get_steps(behavior_name)
         decision_steps for agents not in terminal state, as list of:
         - an observation of the last steps
         - the reward as its result in float
         - an int of the agent's id as unique identifier
         - an action_mask, only available for multi-discrete actions

         terminal_steps for agents who reached a terminal state, as list of:
         - an observation of the last steps
         - the reward as its result in float
         - "interrupted" as a boolean if the agent was interrupted in the last step
         - an int of the agent's id as unique identifier
        """
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        
        # for every agent who requires a new decision
        for decision_agent_id in decision_steps.agent_id:
            
            dec_agent_idx = decision_steps.agent_id_to_index[decision_agent_id]   

            # if agent has no past observation
            if decision_agent_id not in self.trajectory_by_agent.keys():
                self.trajectory_by_agent[decision_agent_id] = []
                self.cumulative_reward_by_agent[decision_agent_id] = 0
            
            # if agent has already had past observation
            else:                
                # create a new experience based on the last stored info and the new info
                new_experience = Experience(
                    obs=self.last_observation_by_agent[decision_agent_id].copy(), 
                    action=self.last_action_by_agent[decision_agent_id].copy(), 
                    reward=decision_steps.reward[dec_agent_idx].copy(),
                    done=False,
                    next_obs=decision_steps.obs[0][dec_agent_idx].copy()
                )
                # add to the tuple of experiences to be returned
                experiences.append(new_experience)
                # add to trajectory and update cumulative reward
                self.trajectory_by_agent[decision_agent_id].append(new_experience)
                self.cumulative_reward_by_agent[decision_agent_id] += decision_steps.reward[dec_agent_idx]
            # update last observation
            self.last_observation_by_agent[decision_agent_id] = decision_steps.obs[0][dec_agent_idx].copy()

        # for every agent which has reached a terminal state
        for terminal_agent_id in terminal_steps.agent_id:

            term_agent_idx = terminal_steps.agent_id_to_index[terminal_agent_id]

            # if agent has no past observation, policy is not used in anyway, so we pass
            if terminal_agent_id not in self.trajectory_by_agent.keys(): pass

            # if agent has past observation
            else:
                # create the last experience based on the last stored info and the new info
                last_experience = Experience(
                    obs=self.last_observation_by_agent[terminal_agent_id].copy(), 
                    action=self.last_action_by_agent[terminal_agent_id].copy(), 
                    reward=terminal_steps.reward[term_agent_idx],
                    done=not terminal_steps.interrupted[term_agent_idx],
                    next_obs=terminal_steps.obs[0][term_agent_idx].copy()
                )
                # add to the tuple for experiences to be returned
                experiences.append(last_experience)
                # update trajectory and cumulative reward
                self.trajectory_by_agent[terminal_agent_id].append(last_experience)
                self.cumulative_reward_by_agent[terminal_agent_id] += terminal_steps.reward[term_agent_idx]
                # move trajectory to buffer and report cumulative reward while removing them from dict
                terminal_trajectories.append(self.trajectory_by_agent.pop(terminal_agent_id))
                terminal_cumulative_rewards.append(self.cumulative_reward_by_agent.pop(terminal_agent_id))
                # remove last action and observation from dicts
                self.last_observation_by_agent.pop(terminal_agent_id)
                self.last_action_by_agent.pop(terminal_agent_id)

        # if there still are agents requesting new actions
        if len(decision_steps.obs[0]) != 0:
            # Generate actions for agents while applying the exploration function to 
            # promote exploration of the world
            
            best_actions = exploration_function(
                learning_algorithm(decision_steps.obs[0]), 
                self.env
            )
            
            # Store info of action picked to generate new experience in the next loop
            for agent_idx, agent_id in enumerate(decision_steps.agent_id):
                self.last_action_by_agent[agent_id] = best_actions[agent_idx]
            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            
            action_tuple = ActionTuple()
            if self.env.behavior_specs[behavior_name].action_spec.is_continuous:
                
                action_tuple.add_continuous(best_actions)
            else:
                action_tuple.add_discrete(best_actions)
            self.env.set_actions(behavior_name, action_tuple)
            
        # Perform a step in the simulation
        self.env.step()

        return experiences, terminal_trajectories, terminal_cumulative_rewards
    
    # TODO A function which trains the given algorithm through on-policy learning.
    def train(self):
        return


class UnityOffPolicyBaseTrainer:

    BUFFER_IMPLEMENTATION = NdArrayBuffer
    
    def __init__(
            self, 
            env : BaseEnv, 
            behavior_name : BehaviorName
        ):
        """
        Creates an off policy base trainer with a given BaseEnv and BehaviorName specifying agent.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param BaseEnv env: The Unity environment used.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        """
        if not isinstance(env, BaseEnv):
            raise Exception("The environment passed to UnityOffPolicyBaseTrainer was not a Unity " + 
                            "environment - please make sure that a Unity environment is passed!")

        self.env = env
        self.behavior_name = behavior_name

        # reset the environment to produce behavior_specs
        self.env.reset()
        self.observation_shape = self.env.behavior_specs[behavior_name].observation_specs[0].shape
        self.action_shape = (self.env.behavior_specs[behavior_name].action_spec.continuous_size, )
        
        self.cumulative_reward_by_agent : Dict[int, float] = {}
        self.last_action_by_agent : Dict[int, np.ndarray] = {}
        self.last_observation_by_agent : Dict[int, np.ndarray] = {}

        self.terminal_cumulative_rewards : List = []
        
        print("The environment has observation of size: ", self.observation_shape[0],
            " and action of size: ", self.action_shape[0], ".")


    def generate_experience(
        self,
        exploration_function,
        learning_algorithm,
        behavior_name : BehaviorName=None
        ) -> Buffer:
        """
        Executes a single step in the environment for every agent with behavior_name and
        according to policy by learning_algorithm, storing the last state & action, 
        trajectory so far, and cumulative rewards for each agent. 
        Returns the experience of agents in the last step, and trajectories & cumulative rewards
        for agents who reached a terminal step.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :param BehaviorName behavior_name: The behavior_name of the agent for which we 
        generate the experience.
        :returns Buffer experiences: The experiences of every agent that took action in
        the last step.
        """
        if behavior_name is None: behavior_name = self.behavior_name
        """
        Generating one batch of trajectory comes in the following loop
        1) Get info from environment via get_steps(behavior_name)
         decision_steps for agents not in terminal state, as list of:
         - an observation of the last steps
         - the reward as its result in float
         - an int of the agent's id as unique identifier
         - an action_mask, only available for multi-discrete actions

         terminal_steps for agents who reached a terminal state, as list of:
         - an observation of the last steps
         - the reward as its result in float
         - "interrupted" as a boolean if the agent was interrupted in the last step
         - an int of the agent's id as unique identifier
        """
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)

        # instantiate a buffer with size equal to decision_steps.agent_id.size, which is equal to how many new experience we'll have
        experiences : Buffer = UnityOffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(
            max_size=decision_steps.agent_id.size + terminal_steps.agent_id.size,
            obs_shape=self.observation_shape,
            act_shape=self.action_shape
            )
        
        # for every agent who requires a new decision
        for decision_agent_id in decision_steps.agent_id:
            
            dec_agent_idx = decision_steps.agent_id_to_index[decision_agent_id]   

            # if agent has no past observation
            if decision_agent_id not in self.last_observation_by_agent.keys():
                self.cumulative_reward_by_agent[decision_agent_id] = 0
            # if agent already has past observation 
            else:    
                # create a new experience based on the last stored info and the new info
                new_experience = Experience(
                    obs=self.last_observation_by_agent[decision_agent_id].copy(), 
                    action=self.last_action_by_agent[decision_agent_id].copy(), 
                    reward=decision_steps.reward[dec_agent_idx].copy(),
                    done=False,
                    next_obs=decision_steps.obs[0][dec_agent_idx].copy()
                )
                # add to the tuple of experiences to be returned
                experiences.append_experience(new_experience.obs, new_experience.action, 
                                              new_experience.reward, new_experience.done, 
                                              new_experience.next_obs)
                # update cumulative reward
                self.cumulative_reward_by_agent[decision_agent_id] += decision_steps.reward[dec_agent_idx]

            # update last observation
            self.last_observation_by_agent[decision_agent_id] = decision_steps.obs[0][dec_agent_idx].copy()

        # for every agent which has reached a terminal state
        for terminal_agent_id in terminal_steps.agent_id:

            term_agent_idx = terminal_steps.agent_id_to_index[terminal_agent_id]

            # if agent has no past observation, policy is not used in anyway, so we pass
            if terminal_agent_id not in self.last_observation_by_agent.keys(): pass

            # if agent has past observation
            else:
                # create the last experience based on the last stored info and the new info
                last_experience = Experience(
                    obs=self.last_observation_by_agent[terminal_agent_id].copy(), 
                    action=self.last_action_by_agent[terminal_agent_id].copy(), 
                    reward=terminal_steps.reward[term_agent_idx],
                    done=not terminal_steps.interrupted[term_agent_idx],
                    next_obs=terminal_steps.obs[0][term_agent_idx].copy()
                )
                # add to the tuple for experiences to be returned
                experiences.append_experience(last_experience.obs, last_experience.action, 
                                              last_experience.reward, last_experience.done, 
                                              last_experience.next_obs)
                # update cumulative reward
                self.cumulative_reward_by_agent[terminal_agent_id] += terminal_steps.reward[term_agent_idx]
                # add this terminal cumulative reward to be tracked
                self.terminal_cumulative_rewards.append(self.cumulative_reward_by_agent[terminal_agent_id])
                # remove last action and observation from dicts
                self.last_observation_by_agent.pop(terminal_agent_id)
                self.last_action_by_agent.pop(terminal_agent_id)

        # if there still are agents requesting new actions
        if len(decision_steps.obs[0]) != 0:
            # Generate actions for agents while applying the exploration function to 
            # promote exploration of the world
            
            best_actions = exploration_function(
                learning_algorithm(decision_steps.obs[0]), 
                self.env
            )
            
            # Store info of action picked to generate new experience in the next loop
            for agent_idx, agent_id in enumerate(decision_steps.agent_id):
                self.last_action_by_agent[agent_id] = best_actions[agent_idx]
            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            
            action_tuple = ActionTuple()
            if self.env.behavior_specs[behavior_name].action_spec.is_continuous:
                
                action_tuple.add_continuous(best_actions)
            else:
                action_tuple.add_discrete(best_actions)
            self.env.set_actions(behavior_name, action_tuple)
            
        # Perform a step in the simulation
        self.env.step()

        return experiences
    
    def generate_batch_of_experiences(
            self, 
            buffer_size : int,
            exploration_function,
            learning_algorithm,
            behavior_name : BehaviorName
        ) -> Buffer:
        """
        Generates and returns a buffer containing "buffer_size" random experiences 
        sampled from running the policy from learning_algorithm in the environment.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.

        :param int buffer_size: The size of the buffer to be returned.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experience.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :param BehaviorName behavior_name: The behavior_name of the agent for which we 
        generate the experience.
        :returns Buffer buffer: The buffer containing buffer_size experiences.
        """
        # Create an empty buffer
        buffer : Buffer = UnityOffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=buffer_size,
                                                                               obs_shape=self.observation_shape,
                                                                               act_shape=self.action_shape)
        
        # While we have not enough experience in buffer, generate experience
        while buffer.size() < buffer_size:
            new_experience = self.generate_experience(
                exploration_function=exploration_function,
                learning_algorithm=learning_algorithm,
                behavior_name=behavior_name
            )
            
            buffer.extend_buffer(new_experience)

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
            task_name : str
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
        """
        if not isinstance(learning_algorithm, PolicyLearningAlgorithm):
            raise Exception("The algorithm passed to train was not an instance of " + 
                            "OffPolicyLearningAlgorithm - please make sure that an " +
                            "OffPolicyLearningAlgorithm is passed!")

        start_time = timer()

        # Reset env
        self.env.reset()

        try:
            experiences : Buffer = UnityOffPolicyBaseTrainer.BUFFER_IMPLEMENTATION(max_size=max_buffer_size,
                                                                                   obs_shape=self.observation_shape,
                                                                                   act_shape=self.action_shape)
            cumulative_rewards: List[float] = []

            # first populate the buffer with num_initial_experiences experiences
            print(f"Generating {num_initial_experiences} initial experiences...")
            init_exp = self.generate_batch_of_experiences(
                buffer_size=num_initial_experiences,
                exploration_function=initial_exploration_function,
                learning_algorithm=learning_algorithm,
                behavior_name=self.behavior_name
            )
            print("Generation successful!")

            experiences.extend_buffer(init_exp)
                
            # then go into the training loop
            for i in range(num_training_steps):
                
                new_exp = self.generate_batch_of_experiences(
                    buffer_size=num_new_experience,
                    exploration_function=training_exploration_function,
                    learning_algorithm=learning_algorithm,
                    behavior_name=self.behavior_name
                )
                
                experiences.extend_buffer(new_exp)
                learning_algorithm.update(experiences)
                
                # evaluate sometimes
                if (i + 1) % (evaluate_every_N_steps) == 0:
                    cumulative_reward = self.evaluate(learning_algorithm, num_samples=3)
                    cumulative_rewards.append(cumulative_reward)
                    print(f"Training loop {i+1} successfully ended: reward={cumulative_reward}.\n")
                
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
            print("Successfully closed env!")

            print(f"Execution time: {end_time - start_time} sec.") #float fractional seconds?

            # Show the training graph
            try:
                plt.clf()
                plt.plot(range(0, len(cumulative_rewards)*evaluate_every_N_steps, evaluate_every_N_steps), cumulative_rewards)
                plt.savefig(f"{task_name}_cumulative_reward_fig.png")
                plt.show()
                learning_algorithm.show_loss_history(task_name=task_name, save_figure=True, save_dir=None)
            except Exception:
                logging.error(traceback.format_exc())
                
            return learning_algorithm
    
    def evaluate(self, 
                 learning_algorithm : PolicyLearningAlgorithm, 
                 num_samples : int = 5) -> float:
        """
        Returns the average of the last num_samples terminal cumulative rewards as 
        measurement of algorithm performance.

        :param OffPolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
        evaluating actions given states per batches.
        :param int num_samples: The number of cumulative rewards we average. Defaults to 5.
        :returns float: Returns the average cumulative reward.
        """
        if not isinstance(learning_algorithm, PolicyLearningAlgorithm):
            raise Exception("The algorithm passed to UnityOffPolicyBaseTrainer was not an instance of " + 
                            "PolicyLearningAlgorithm - please make sure that an " +
                            "PolicyLearningAlgorithm is passed!")
        
        cumulative_reward = sum(self.terminal_cumulative_rewards[-num_samples:]) / num_samples
        return cumulative_reward
