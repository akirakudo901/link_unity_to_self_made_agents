"""
Base trainer object that one can use to interact with the grid world unity environment under the
mlagents_envs format.
Allows to interact with the environment and to store information through the interaction.

_______________________________________________________
A bulk of the code was taken from the Trainer object in the Colab tutorial found by 
following the link below, then to section "Python Tutorial with Google Colab", 
then to "Q-Learning with a UnityEnvironment".

https://unity-technologies.github.io/ml-agents/ML-Agents-Toolkit-Documentation/

Shout out! Only slightly modified below.

"""

from typing import Dict, List, NamedTuple

from mlagents_envs.environment import ActionTuple, BaseEnv
import numpy as np

"""
Except the base trainer, we can also define classes that ensure type settings and
can come to handy.
"""

class GridWorldExperience(NamedTuple):
  """
  An experience in the grid world containing the data of one Agent transition.
  - Observation
  - Action
  - Reward
  - Done flag
  - Next Observation
  """

  obs: np.ndarray
  action: np.ndarray
  reward: float
  done: bool
  next_obs: np.ndarray

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[GridWorldExperience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[GridWorldExperience]

class GridWorldBaseTrainer:
    @staticmethod
    def generate_trajectories(
        env: BaseEnv, learning_algo, buffer_size: int, epsilon: float
    ):
        """
        Given a Unity Environment and a learning algorithm, this method will generate a
        buffer of Experiences obtained by running the Environment with the Policy
        derived from the algorithm.
        :param BaseEnv: The UnityEnvironment used.
        :param learning_algo: The algorithm used to collect the data. Takes in a numpy array
        of observation, and returns a numpy array of values evaluating utility of all actions
        :param buffer_size: The minimum size of the buffer this method will return.
        :param epsilon: Will add a random normal variable with standard deviation.
        epsilon to the value heads of the Q-Network to encourage exploration.
        :returns: a Tuple containing the created buffer and the average cumulative
        the Agents obtained.
        """
        # Create an empty Buffer
        buffer: Buffer = []

        # Reset the environment
        env.reset()
        # Read and store the Behavior Name of the Environment
        behavior_name = list(env.behavior_specs)[0]
        # Read and store the Behavior Specs of the Environment
        spec = env.behavior_specs[behavior_name]

        # Create a Mapping from AgentId to Trajectories. This will help us create
        # trajectories for each Agents
        dict_trajectories_from_agent: Dict[int, Trajectory] = {}
        # Create a Mapping from AgentId to the last observation of the Agent
        dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to the last observation of the Agent
        dict_last_action_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to cumulative reward (Only for reporting)
        dict_cumulative_reward_from_agent: Dict[int, float] = {}
        # Create a list to store the cumulative rewards obtained so far
        cumulative_rewards: List[float] = []

        while len(buffer) < buffer_size:  # While not enough data in the buffer
            # Get the Decision Steps and Terminal Steps of the Agents
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # permute the tensor to go from NHWC to NCHW
            order = (0, 3, 1, 2)
            decision_steps.obs = [np.transpose(
                obs, order) for obs in decision_steps.obs]
            terminal_steps.obs = [np.transpose(
                obs, order) for obs in terminal_steps.obs]

            # For all Agents with a Terminal Step:
            for agent_id_terminated in terminal_steps:
                # Create its last experience (is last because the Agent terminated)
                last_experience = GridWorldExperience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(
                    ),
                    next_obs=terminal_steps[agent_id_terminated].obs[0],
                )
                # Clear its last observation and action (Since the trajectory is over)
                dict_last_obs_from_agent.pop(agent_id_terminated)
                dict_last_action_from_agent.pop(agent_id_terminated)
                # Report the cumulative reward
                cumulative_reward = (
                    dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                    + terminal_steps[agent_id_terminated].reward
                )
                cumulative_rewards.append(cumulative_reward)
                # Add the Trajectory and the last experience to the buffer
                buffer.extend(dict_trajectories_from_agent.pop(
                    agent_id_terminated))
                buffer.append(last_experience)

            # For all Agents with a Decision Step:
            for agent_id_decisions in decision_steps:
                # If the Agent does not have a Trajectory, create an empty one
                if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = 0

                # If the Agent requesting a decision has a "last observation"
                if agent_id_decisions in dict_last_obs_from_agent:
                    # Create an Experience from the last observation and the Decision Step
                    exp = GridWorldExperience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(
                        ),
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(
                        ),
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                    )
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(
                        exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                # Store the observation as the new "last observation"
                dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs[0]
                )

            # Generate an action for all the Agents that requested a decision
            # Compute the values for each action given the observation
            actions_values = (
                learning_algo(decision_steps.obs[0])
            )
            # Add some noise with epsilon to the values
            actions_values += epsilon * (
                np.random.randn(
                    actions_values.shape[0], actions_values.shape[1])
            ).astype(np.float32)
            # Pick the best action using argmax
            actions = np.argmax(actions_values, axis=1)
            actions.resize((len(decision_steps), 1))
            # Store the action that was picked, it will be put in the trajectory later
            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index]

            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            env.set_actions(behavior_name, action_tuple)
            # Perform a step in the simulation
            env.step()
        return buffer, np.mean(cumulative_rewards)
