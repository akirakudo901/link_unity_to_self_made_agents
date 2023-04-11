"""
The general flow I got from reading the example:
-) the environment interacts with this API via:
1 - setting up the environment somehow
2 - requesting a step in the environment 
3 - obtaining info from the environment to be used for training the agent, namely:
 = the new observation
 = reward as result of the action
 = whether the state is terminal or not (which determines how we use the data to train our agent)
 to be all stored as pairs of "Experience" objects:
 = old observation
 = action taken
 = reward as a result of action
 = new observation
 = if the new state was terminal or not
4 - creating a new set of actions based on the policy, to be returned to the environment 

I ACTUALLY NEED TO DO A CONVERSION OF THE WORLD'S DESCRIPTION INTO A DISCRETE STATE
TABLE THAT I CAN STORE AS VALUES INTO OUR QTABLE.
THIS PART MIGHT REQUIRE ME TO MAKE A NETWORK PURELY FOR THAT.
COOL STUFF HAPPENING, DAMN!

"""

from typing import NamedTuple, List, Dict

from mlagents_envs.environment import UnityEnvironment, BaseEnv
import numpy as np

import qtable

class Experience(NamedTuple):
    """
    An experience contains the data of one Agent transition.
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
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]



class Trainer:
    @staticmethod
    def generate_trajectories(
    env: BaseEnv, qtable: qtable.Qtable, buffer_size: int, epsilon: float
    ):
        """
        Given a Unity Environment and a Q-Network, this method will generate a
        buffer of Experiences obtained by running the Environment with the Policy
        derived from the Q-Network.
        :param BaseEnv: The UnityEnvironment used.
        :param q_table: The Q-Table used by the agent to make decisions.
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
        # Create a Mapping from AgentId to the last action of the Agent
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
            decision_steps.obs = [np.transpose(obs, order) for obs in decision_steps.obs]
            terminal_steps.obs = [np.transpose(obs, order) for obs in terminal_steps.obs]

            # For all Agents with a Terminal Step:
            for agent_id_terminated in terminal_steps:
                # Create its last experience (is last because the Agent terminated)
                last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(),
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
                buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
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
                    exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                    )
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )

                # Store the observation as the new "last observation"
                dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs[0]
                )

            # Generate an action for all the Agents that requested a decision
            # Compute the values for each action given the observation

            # HERE COMES THE FIRST MAJOR CHANGE SINCE WE ARE USING A QTABLE.
            # I RETURNED THE REWARDS OBTAINED BY TAKING EACH DECISIONS.
            act_vals = []
            for img_obs in decision_steps.obs[0]:
                disc_obs = CONVERT_IMAGEOBS_TO_DISCRETE(img_obs)
                act_vals.append(qtable.get_optimal_action(disc_obs))
                
            actions_values = np.array(act_vals)

            #ANOTHER CHANGE: INTERWEAVING SOME RANDOM ACTION IN RELATION TO EPSILON
            # Add some noise with epsilon to the values
            actions_values += epsilon * (
                np.random.randn(actions_values.shape[0], actions_values.shape[1])
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

    @staticmethod
    def update_q_net(
        q_net: VisualQNetwork,
        optimizer: torch.optim,
        buffer: Buffer,
        action_size: int
    ):
        """
        Performs an update of the Q-Network using the provided optimizer and buffer
        """
        BATCH_SIZE = 1000
        NUM_EPOCH = 3
        GAMMA = 0.9
        batch_size = min(len(buffer), BATCH_SIZE)
        random.shuffle(buffer)
        # Split the buffer into batches
        batches = [
        buffer[batch_size * start : batch_size * (start + 1)]
        for start in range(int(len(buffer) / batch_size))
        ]
        for _ in range(NUM_EPOCH):
        for batch in batches:
            # Create the Tensors that will be fed in the network
            obs = torch.from_numpy(np.stack([ex.obs for ex in batch]))
            reward = torch.from_numpy(
            np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
            )
            done = torch.from_numpy(
            np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1)
            )
            action = torch.from_numpy(np.stack([ex.action for ex in batch]))
            next_obs = torch.from_numpy(np.stack([ex.next_obs for ex in batch]))

            # Use the Bellman equation to update the Q-Network
            target = (
            reward
            + (1.0 - done)
            * GAMMA
            * torch.max(q_net(next_obs).detach(), dim=1, keepdim=True).values
            )
            mask = torch.zeros((len(batch), action_size))
            mask.scatter_(1, action, 1)
            prediction = torch.sum(q_net(obs) * mask, dim=1, keepdim=True)
            criterion = torch.nn.MSELoss()
            loss = criterion(prediction, target)

            # Perform the backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
