"""
Represents a base trainer which can be adapted to any unity environment
by making a trainer which inherits from it.

~Classes~
 Experience:  a template for a single transition experience in the given world
 BaseTrainer: a handler of interaction between the learning algorithm and the unity environment
              to collect experience and update the algorithm
"""

import random
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from mlagents_envs.environment import ActionTuple, BaseEnv, BehaviorName
import matplotlib.pyplot as plt
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


class OnPolicyBaseTrainer:

    def __init__(
            self, 
            env : BaseEnv, 
            behavior_name : BehaviorName,
            learning_algorithm
        ):
        """
        Creates an on policy base trainer with a given BaseEnv, BehaviorName specifying 
        agent and a learning algorithm which holds a policy.
        Enables the generation of the experience for a single time step and return it
        to be used for on-policy learning. 
        TODO! (shall I include an entry determining how exploration is handled?)

        :param BaseEnv env: The Unity environment used.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        """
        self.env = env
        self.behavior_name = behavior_name
        self.learning_algorithm = learning_algorithm

        self.trajectory_by_agent : Dict[int, Trajectory] = {}
        self.cumulative_reward_by_agent : Dict[int, float] = {}
        self.last_action_by_agent : Dict[int, np.ndarray] = {}
        self.last_observation_by_agent : Dict[int, np.ndarray] = {}

    def generate_experience(
        self,
        exploration_function,
        epsilon : float,
        behavior_name : BehaviorName=None
        ) -> Tuple[Buffer, List[Trajectory], List[float]]:
        """
        Executes a single step in the environment for every agent with behavior_name, 
        storing the last state & action, trajectory so far, and cumulative rewards for 
        each agent. 
        Returns the experience of agents in the last step, and trajectories & cumulative rewards
        for agents who reached a terminal step.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by exploration_function and epsilon.
        :param exploration_function: A function which together with epsilon, controls 
        how exploration is handled during collection of experiences.
        :param float epsilon: Value epsilon which specifies exploration together with
        exploration_funcction.
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
                self.learning_algorithm(decision_steps.obs[0]), 
                epsilon,
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


class OffPolicyBaseTrainer(OnPolicyBaseTrainer):

    def __init__(
            self, 
            env : BaseEnv, 
            behavior_name : BehaviorName,
            learning_algorithm
        ):
        """
        Creates an off policy trainer with a given BaseEnv, BehaviorName specifying 
        agent and a learning algorithm which holds a policy. Can be used to generate
        batches of experience to be used for off-policy learning.
        TODO! (shall I include an entry determining how exploration is handled?)

        :param BaseEnv env: The Unity environment used.
        :param BehaviorName behavior_name: The BehaviorName of interest 
         *see low level Python API documentation for details on BehaviorName
        :param learning_algorithm: The algorithm which provides policy, evaluating 
        actions given states per batches.
        """
        super().__init__(env, behavior_name, learning_algorithm)

    def generate_batch_of_experiences(
            self, 
            buffer_size : int,
            exploration_function,
            epsilon : float,
            behavior_name : BehaviorName
        ) -> Tuple[Buffer, float]:
        """
        Generates and returns a buffer containing "buffer_size" random experiences 
        sampled from running the policy in the environment.
        The experience involves some exploration (vs. exploitation) which behavior 
        is determined by epsilon. 
        TODO! (shall I include some terms determining how epsilon is modified or handled?)

        :param int buffer_size: The size of the buffer to be returned.
        :param exploration_function: A function which, together with epsilon, controls how 
        exploration is handled during collection of experience.
        :param float epsilon: Value epsilon which specifies exploration together with
        exploration_funcction.
        :param BehaviorName behavior_name: The behavior_name of the agent for which we 
        generate the experience.
        :returns Buffer buffer: The buffer containing buffer_size experiences.
        :returns float average_cumulative_reward: The average reward for agents over the entire
        trajectories in the buffer.
        """
        # Create an empty buffer and list of rewards to be returned
        buffer : Buffer = []
        cumulative_reward : List[float] = []

        # While we have not enough experience in buffer, generate experience
        while len(buffer) < buffer_size:
            _, new_trajectories, new_cumulative_rewards = self.generate_experience(
                exploration_function=exploration_function,
                epsilon=epsilon,
                behavior_name=behavior_name
            )
            buffer.extend([experience for trajectory in new_trajectories for experience in trajectory])
            cumulative_reward.extend(new_cumulative_rewards)
        # Cut down the number of experience to buffer_size
        buffer = buffer[:buffer_size]

        return buffer, np.mean(cumulative_reward)
    
    def train(
            self,
            num_training_steps : int,
            num_new_experience : int,
            max_buffer_size : int,
            exploration_function,
            epsilon : float,
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
        :param exploration_function: A function which together with epsilon, controls 
        how exploration is handled during collection of experiences.
        :param float epsilon: The value of epsilon to control exploration vs. exploitation.
        :param bool save_after_training: Whether to save the policy after training.
        :param str task_name: The name of the task to log when saving after training.
        """
        # Reset env
        self.env.reset()

        try:
            
            cumulative_rewards: List[float] = []
            experiences : Buffer = []

            for _ in tqdm(range(num_training_steps)):
                new_exp, _ = self.generate_batch_of_experiences(
                    buffer_size=num_new_experience,
                    exploration_function=exploration_function,
                    epsilon=epsilon,
                    behavior_name=self.behavior_name
                )
                experiences.extend(new_exp)
                random.shuffle(experiences)
                if len(experiences) > max_buffer_size:
                    experiences = experiences[:max_buffer_size]
                self.learning_algorithm.update(experiences)

                _, reward = self.generate_batch_of_experiences(
                    buffer_size=100,
                    exploration_function=exploration_function,
                    epsilon=0,
                    behavior_name=self.behavior_name
                )
                cumulative_rewards.append(reward)

            if save_after_training: self.learning_algorithm.save(task_name)

            return self.learning_algorithm

        except KeyboardInterrupt:
            print("\nTraining interrupted, continue to next cell to save to save the model.")
        finally:
            self.env.close()

        # Show the training graph
        try:
            plt.plot(range(num_training_steps), cumulative_rewards)
            plt.show()
        except ValueError:
            print("\nPlot failed on interrupted training.")