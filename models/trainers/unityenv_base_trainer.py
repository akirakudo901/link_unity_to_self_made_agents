"""
Represents a base trainer which can be adapted to any unity environment
by making a trainer which inherits from it.

~Classes~
 Experience:  a template for a single transition experience in the given world
 BaseTrainer: a handler of interaction between the learning algorithm and the unity environment
              to collect experience and update the algorithm
"""

from typing import Dict, List, Tuple

import numpy as np

from mlagents_envs.environment import ActionTuple, BaseEnv, BehaviorName

from models.trainers.base_trainer import OffPolicyBaseTrainer
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
            env_name : str,
            behavior_name : BehaviorName
        ):
        """
        Creates an on policy base trainer with a given BaseEnv and BehaviorName specifying agent.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param BaseEnv env: The Unity environment used.
        :param str env_name: The name of the given environment.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        """
        self.env = env
        self.env_name = env_name
        self.behavior_name = behavior_name

        self.trajectory_by_agent : Dict[int, Trajectory] = {}
        self.cumulative_reward_by_agent : Dict[int, float] = {}
        self.last_action_by_agent : Dict[int, np.ndarray] = {}
        self.last_observation_by_agent : Dict[int, np.ndarray] = {}

    def generate_experience(
        self,
        exploration_function,
        learning_algorithm
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
        :returns Buffer experiences: The experiences of every agent that took action in
        the last step.
        :returns List[Trajectory] terminal_trajectories: The list of trajectories for agents
        that reached a terminal state.
        :returns List[float] terminal_cumulative_rewards: The list of rewards for agents that 
        reached a terminal state. 
        """
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
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
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
                        
            best_actions = exploration_function(decision_steps.obs[0])
            
            # Store info of action picked to generate new experience in the next loop
            for agent_idx, agent_id in enumerate(decision_steps.agent_id):
                self.last_action_by_agent[agent_id] = best_actions[agent_idx]
            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            
            action_tuple = ActionTuple()
            if self.env.behavior_specs[self.behavior_name].action_spec.is_continuous:
                
                action_tuple.add_continuous(best_actions)
            else:
                action_tuple.add_discrete(best_actions)
            self.env.set_actions(self.behavior_name, action_tuple)
            
        # Perform a step in the simulation
        self.env.step()

        return experiences, terminal_trajectories, terminal_cumulative_rewards
    
    # TODO A function which trains the given algorithm through on-policy learning.
    def train(self):
        return


class UnityOffPolicyBaseTrainer(OffPolicyBaseTrainer):
    
    def __init__(
            self,
            env : BaseEnv, 
            env_name : str,
            behavior_name : BehaviorName
        ):
        """
        Creates an off policy base trainer with a given BaseEnv and BehaviorName specifying agent.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning.

        :param BaseEnv env: The Unity environment used.
        :param str env_name: The name of the given environment.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        """
        if not isinstance(env, BaseEnv):
            raise Exception("The environment passed to UnityOffPolicyBaseTrainer was not a Unity " + 
                            "environment - please make sure that a Unity environment is passed!")
        
        super().__init__(env=env)
        self.env_name = env_name
        self.behavior_name = behavior_name

        # reset the environment to produce behavior_specs
        self.env.reset()
        self.obs_shape = self.env.behavior_specs[behavior_name].observation_specs[0].shape
        self.act_shape = (self.env.behavior_specs[behavior_name].action_spec.continuous_size, )
        
        self.cumulative_reward_by_agent : Dict[int, float] = {}
        self.last_action_by_agent : Dict[int, np.ndarray] = {}
        self.last_observation_by_agent : Dict[int, np.ndarray] = {}

        self.terminal_cumulative_rewards : List = []
        
        print("The environment has observation of size: ", self.obs_shape[0],
            " and action of size: ", self.act_shape[0], ".")


    def generate_experience(
        self,
        buffer : Buffer,
        exploration_function,
        learning_algorithm
        ):
        """
        Executes a single step in the environment for every agent with behavior_name and
        according to policy by learning_algorithm, storing the last state & action, 
        trajectory so far, and cumulative rewards for each agent. 
        Adds the experience to the buffer passed.
        Also returns the number of experiences generated.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function.

        :param Buffer buffer: The buffer to which we add experiences.
        :param exploration_function: A function which controls how exploration is handled 
        during collection of experiences.
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        :returns int: The number of experiences newly generated.
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        # for every agent who requires a new decision
        for decision_agent_id in decision_steps.agent_id:
            
            dec_agent_idx = decision_steps.agent_id_to_index[decision_agent_id]   

            # if agent has no past observation
            if decision_agent_id not in self.last_observation_by_agent.keys():
                self.cumulative_reward_by_agent[decision_agent_id] = 0
            # if agent already has past observation 
            else:    
                # create a new experience based on the last stored info and the new info
                # adding it into the buffer
                buffer.append_experience(obs=self.last_observation_by_agent[decision_agent_id].copy(),
                                         act=self.last_action_by_agent[decision_agent_id].copy(),
                                         rew=decision_steps.reward[dec_agent_idx].copy(),
                                         don=False,
                                         next_obs=decision_steps.obs[0][dec_agent_idx].copy())
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
                # adding it to the buffer
                buffer.append_experience(obs=self.last_observation_by_agent[terminal_agent_id].copy(), 
                                         act=self.last_action_by_agent[terminal_agent_id].copy(), 
                                         rew=terminal_steps.reward[term_agent_idx],
                                         don=not terminal_steps.interrupted[term_agent_idx],
                                         next_obs=terminal_steps.obs[0][term_agent_idx].copy())
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
            
            best_actions = exploration_function(decision_steps.obs[0])
            
            # Store info of action picked to generate new experience in the next loop
            for agent_idx, agent_id in enumerate(decision_steps.agent_id):
                self.last_action_by_agent[agent_id] = best_actions[agent_idx]
            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            
            action_tuple = ActionTuple()
            if self.env.behavior_specs[self.behavior_name].action_spec.is_continuous:
                
                action_tuple.add_continuous(best_actions)
            else:
                action_tuple.add_discrete(best_actions)
            self.env.set_actions(self.behavior_name, action_tuple)
            
        # Perform a step in the simulation
        self.env.step()

        return (decision_steps.agent_id.size + terminal_steps.agent_id.size)
    
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
        Returns the average of the last num_samples terminal cumulative rewards as 
        measurement of algorithm performance.

        :param OffPolicyLearningAlgorithm learning_algorithm: The algorithm which provides policy, 
        evaluating actions given states per batches.
        :param int num_samples: The number of cumulative rewards we average. Defaults to 5.
        :returns float: Returns the average cumulative reward.
        """
        self.terminal_cumulative_rewards = self.terminal_cumulative_rewards[-num_samples:]
        
        cumulative_reward = sum(self.terminal_cumulative_rewards) / num_samples
        return cumulative_reward
