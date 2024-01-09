"""Wraps a Unity environment to be used as a dm_env environment."""

import string
from tokenize import String
from typing import Any, Dict, List, Optional, Union
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import torch as th

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
import numpy as np



class UnityWrapper:
    """Environment wrapper for Unity Environments."""

    def __init__(
        self,
        # file_name: string,
        environment: UnityEnvironment,
        episode_limit: int = 1500,
        config = None
        ):
        super(UnityWrapper, self).__init__()
        self._environment = environment
        

        if config is not None:
            self.config = config


        self._environment.reset()
        self._behaviour_name = list(self._environment.behavior_specs)[0]
        self._behaviour_specs = self._environment.behavior_specs[self._behaviour_name]
        self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)
        self.step_count = 0
        self.all_agents_byname = self.possible_agents

        # Used during generation of ActionTuple for UnityEnv
        self._action_space = "Discrete"
        self.continuous_action = None
        self.episode_limit = episode_limit

        if not self.config["grayscale"]:
            self.obs_shape = self._behaviour_specs.observation_specs[0].shape
        else:
            self.obs_shape = (self._behaviour_specs.observation_specs[0].shape[0], self._behaviour_specs.observation_specs[0].shape[1], 1)
            # Grayscale coefficients obtained from dubious Google Search
            self.grayscale_coefficients = np.array([0.299, 0.587, 0.114])

        # Set the legal actions masks, for later use
        self._action_mask = None


    def reset(self):
        """
        Resets the episode
        """
        self._environment.reset()
        self.step_count = 0

        self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)
        
        # Have to set this property here, because if called elsewhere, decisionsteps might not have all agents in
        self.all_agents_byname = self.possible_agents



    def step(self, actions = None):
        """"
        Steps the environment once
        Parameters:
            actions: actions for each agent as array of shape (n,1)

        Returns:
            reward: float corresponding to total environment reward
            terminated
            env_info
        """

        if actions is not None:
            self._convert_set_actions(actions)

        # Step the environment, performing the actions passed abover
        self._environment.step()

        # Get the new decisionsteps
        self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)

        # Get the reward from the Unity Environment and convert it to the correct format
        rewards_pc, done = self._get_rewards_dones()


        env_info = self.get_episode_end_reached()
        self.step_count +=1
        

        return rewards_pc, done, env_info

    def get_steps(self):
        return self._environment.get_steps(self._behaviour_name)
    
    def _get_observations(self):
        
        """
        Gets an array of observations in shape (n, w, h, c) where:
            n - number of agents in the environments
            w - width of the observation in pixels
            h - height of the observation in pixels
            c - number of channels - default is 3

        Will always contain the latest observations; from decision steps if the episode has not terminated
        or from terminal steps if the episode has terminated
        """
        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
        else:
            steps_to_use = self.decision_steps
        
        if self.config["grayscale"]:
            return self.convert_to_grayscale(steps_to_use.obs[0])

        return steps_to_use.obs[0]
    
    def convert_to_grayscale(self, obs):
        obs = np.sum(obs*self.grayscale_coefficients, axis=-1, keepdims=True)
        return obs
    
    def _get_global_state_variables(self):
        """
        Gets the observations from the agents that make up part of the global state. This requires the environment itself to be setup to return those values.
        All environments used has the information available, but the information can be omitted during training if you wish to use e.g a concatenated version of
        the agents' observations as the global state. Some sort of global state is required ny the QMIX algorithm, though.
        """

        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
        else:
            steps_to_use = self.decision_steps

        global_state = []
        # Get the agent specific ones first:
        for i,agent in enumerate(self.all_agents_byname):
            # The (2,8) below is hardcoded because there's no way to dynamically determine which part of the vector obs is agent specific and which part is global
            # To be precise: Each agent contains its own position and rotation, as well as the positions of all important world objects
            # The latter is considered global and should be added to the state only once, the rest are agent-specific and should be added to the state for each agent in the system
            for j in range (2,8):
                global_state.append(steps_to_use[i].obs[1][j])
        
        # After this loop, global_state looks something like:
        # [agent_1_x, agent_1_z, agent_1_fwd_x, agent_1_fwd_z, agent_1_right_x, agent_1_right_z, same for agent_2]

        # Now get the rest of the stuff and only add it once, retrieve it from the last value of i sommer
        for j in range(8,steps_to_use[0].obs[1].shape[0]):
            global_state.append(steps_to_use[i].obs[1][j])

        state = np.array(global_state)

        return state


    def _convert_set_actions(self, actions):
        """
        Takes in a tensor of shape (n_agents, 1) for discrete action space
        """

        if self.continuous_action == None:
            self.continuous_action = self._behaviour_specs.action_spec.empty_action(self.get_num_agents()).continuous

        unity_action = ActionTuple()
        if isinstance(actions, th.Tensor):
            actions = actions.detach().cpu().numpy().reshape(-1,1)
        
        unity_action.add_discrete(actions)
        self._environment.set_actions(self._behaviour_name, unity_action)

    
    def _get_rewards_dones(self):

        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
            terminal = True
        else:
            steps_to_use = self.decision_steps
            terminal = False

        # Get agent id-index map. Not strictly necessary but always best to make sure
        agents = steps_to_use.agent_id_to_index

        # Loop through all agents and accumulate reward
        reward = 0
        for _, agent_index in enumerate(agents):
            reward+= steps_to_use[agent_index].reward

        # We want the group reward to be the mean over all agent rewards, so just divide the total reward by the number of agents.
        reward /= self.get_num_agents()

        return reward, terminal


    def env_done(self, terminal_steps) -> bool:
        if len(terminal_steps) > 0:
            return True
        else:
            return False

    def get_avail_actions(self) -> list:
        """
        Get the available/legal actions as a list of one-hot arrays.
        """
        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
        else:
            steps_to_use = self.decision_steps

        legal_actions = []
        for i,agent in enumerate(self.all_agents_byname):
            # The first action should always be legal, as it is the action of "No action"
            legal_actions_list = [1]

            for j in range(2):
                legal_actions_list.append(np.int32(steps_to_use[i].obs[1][j]))

            # The above legal action masking step only masks the first 3 actions, i.e
            # portal placement and no op.
            # Also add masking for all other actions, but make them always true since they are the movement and rotation actions
            if self._action_space == "Discrete":
                # self.action_mask contains a list of all 1's to mask all actions that are always legal.
                if not self._action_mask:
                    length = self.get_num_actions() - len(legal_actions_list)
                    self._action_mask = [1] * length
                fully_masked_actions = legal_actions_list + self._action_mask

            elif self._action_space == "MultiDiscrete":
                raise NotImplementedError


            legal_actions.append(np.array(fully_masked_actions))

        return legal_actions
    
    def get_obs_spec(self):
        """
        Get the shape of observations that the agents return
        """
        return self.obs_shape
    
    def get_num_actions(self):
        """
        Return the number of discrete actions in the environment
        Cases:
            Discrete:
                Returns a single value
        """

        return self._behaviour_specs.action_spec[1][0]
    
    def get_num_agents(self):
        """
        Return the number of agents that is present in the environment
        """
        return len(self.possible_agents)
    
    def get_init_env_info(self):
        """
        PYMARL uses this to help setup replay buffers
        """
        obs_shape = self.get_obs_spec()
        n_actions = self.get_num_actions()
        n_agents = self.get_num_agents()
        
        env_info = {}
        env_info["obs_shape"] = obs_shape
        env_info["n_actions"] = n_actions
        env_info["n_agents"] = n_agents
        env_info["episode_limit"] = self.episode_limit

        # Use only if extra state info is available from the environment - it is by default, but you can choose to not use this information during training
        # This is effectively hard-coded based on mechanics coded into the environments
        # Do not try to change this, as I do not give access to the code for the environments
        if self._behaviour_specs.observation_specs[1].shape[-1]>2:
            # The -2 is because the first 2 values are not part of the global state, the +6 is beacuse each agent's position and rotation is represented by
            # 6 floats, and we have two agents, and only the first ones' values are included by default
            
            state_size_to_add = (len(self.possible_agents)-1)*6 # value is 6

            env_info["state_shape"] = self._behaviour_specs.observation_specs[1].shape[0] - 2 + state_size_to_add

        assert type(env_info["obs_shape"]) == tuple, "Environment is not returning a tuple as observation"

        return env_info
    
    def get_episode_end_reached(self):
        env_info = {}
        if self.step_count == self.episode_limit:
            env_info["episode_limit"] = True
        return env_info

    @property
    def possible_agents(self)->List:
        """All possible agents in env."""

        decision_steps, _ = self._environment.get_steps(self._behaviour_name)
        agents = []

        for i in decision_steps.agent_id:
            agents.append(f"agent_{i}")

        return agents
                