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
import copy

import pdb
import time


# Generating a bunch of pics:
from PIL import Image
import random
import time



class UnityWrapper:
    """Environment wrapper for Unity Environments."""

    def __init__(
        self,
        # file_name: string,
        environment: UnityEnvironment,
        side_channels,
        return_state_info: bool = False,
        env_preprocess_wrappers: Optional[List] = None,
        # encoder: str = None,
        curriculum_steps: list = None,
        use_curriculum_learning: bool = False,
        episode_limit: int = 1500,
        config = None
        ):
        super(UnityWrapper, self).__init__()
        self._environment = environment
        self._reset_next_step = False
        self._return_state_info = return_state_info
        self.previous_obs = {}

        self._use_curriculum_learning = use_curriculum_learning
        self._curriculum_steps = curriculum_steps
        self._current_curriculum_id = 0
        self._side_channels = side_channels

        self.agent0obs = []
        self.agent1obs = []
        self.a0 = 0
        self.a1 = 1
        self.starttime = time.time()-40
        self.generate_gifs = False
        

        if config is not None:
            self.config = config
        # Unity specs are tied to behaviour names, so first get behaviour name and from that, the spec. This code assumes all agents have the same behaviour
        # Reset the original enironment, to generate the behaviour specs and names
        self._environment.reset()
        self._behaviour_name = list(self._environment.behavior_specs)[0]
        self._behaviour_specs = self._environment.behavior_specs[self._behaviour_name]
        self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)
        self.step_count = 0
        self.all_agents_byname = self.possible_agents

        if self._behaviour_specs.action_spec.discrete_size>1:
            self._action_space = "MultiDiscrete" 
            raise NotImplementedError
        else:
            self._action_space = "Discrete"
            self.continuous_action = None
                

        # Temporary position for this
        self.all_agents_byname = self.possible_agents
        # This has the format of a list with contents ['agent_0', 'agent_1', ..., 'agent_n']
        
        # PYMARL requirements:
        self.episode_limit = episode_limit
        # Set some pymarl required stuff:
        if not self.config["grayscale"]:
            self.obs_shape = self._behaviour_specs.observation_specs[0].shape
        else:
            self.obs_shape = (self._behaviour_specs.observation_specs[0].shape[0], self._behaviour_specs.observation_specs[0].shape[1], 1)
            self.grayscale_coefficients = np.array([0.299, 0.587, 0.114])
        self._action_mask = None
        # Set the legal actions masks


        



    def _get_curriculum_steps(self):
        return self._curriculum_steps
        
    def _get_current_curriculum_id(self):
        return self._current_curriculum_id
    
    def _increment_curriculum_id(self):
        self._current_curriculum_id+=1

    def use_curriculum_learning(self):
        return self._use_curriculum_learning
    
    def _send_to_environment(self, parameter):
        self._side_channels[2].set_float_parameter("curriculum_id", parameter)
    
    # def _increase_environment_curriculum(self):s


    def reset(self):
        """
        Resets the episode
        """

        # print("Start execute reset")

        self._reset_next_step = False
        # # self._step_type = dm_env.StepType.FIRST
        # self._step_type = "FIRST"

        # Reset the unity environment
        self._environment.reset()
        self.step_count = 0

        # Use the same decision_steps variable for ALL methods, because the unity environment keeps stepping separately, so repeated calls of the get_steps method
        # will likely lead to extremely unstable training
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


        # set decisionsteps so convert_set_actions works with the latest information 
        # self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)
        # legal_actions = self.get_avail_actions()
        # print(f"Actions taken: {actions}")
        if actions is not None:
            self._convert_set_actions(actions)

        # Step the environment, performing the actions passed abover
        self._environment.step()

        # Get the new decisionsteps
        self.decision_steps, self.terminal_steps = self._environment.get_steps(self._behaviour_name)

        # observation, extras = self._get_observations_dict()
        # observation, extras = self._get_observations_list()
        # observation["legal_actions"] = legal_actions


        # Rewards:
        # Get the reward from the Unity Environment and convert it to the correct format
        # Correct format for rewards is again in the form Dict[agent_id: reward]
        rewards_pc, dones = self._get_rewards_dones()

        env_info = self.get_env_info()
        self.step_count +=1
        

        return rewards_pc, dones, env_info

    def get_steps(self):
        return self._environment.get_steps(self._behaviour_name)
    
    def _get_observations(self):
        
        """
        Gets an array of observations in shape (n, 84, 84, 3) where n is number of agents
        Will always contain the latest observations; from decision steps if the episode has not terminated
        or from terminal steps if the episode has terminated
        """
        # self.decision_steps, self.terminal_steps = self.get_steps()
        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
            # steps_to_use = ts
            # print(f"using teminal at step: {self.step_count}")
        else:
            steps_to_use = self.decision_steps
            # steps_to_use = ds
            # print(f"using decision at step: {self.step_count}")
        
        if self.config["grayscale"]:
            return self.convert_to_grayscale(steps_to_use.obs[0])



        return steps_to_use.obs[0]
    
    def convert_to_grayscale(self, obs):
        obs = np.sum(obs*self.grayscale_coefficients, axis=-1, keepdims=True)
        return obs
    
    def _get_global_state_variables(self):
        """
        Gets the observations from the agents that make up part of the global state. This requires the environment itself to be setup to return those values
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
            # The latter is considered global and should be added to the state only once, the rest are local and should be added to the state for each agent in the system
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
        Takes in a tensor or array of shape (n_agents, 1) for discrete action space
        """

        if self.continuous_action == None:
            self.continuous_action = self._behaviour_specs.action_spec.empty_action(self.get_num_agents()).continuous

        unity_action = ActionTuple()
        if isinstance(actions, th.Tensor):
            # actions = np.array([int(a) for a in actions])
            # actions = np.reshape(actions, newshape = self._behaviour_specs.action_spec.empty_action(self.get_num_agents()).discrete.shape)
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

        # Get agent id-index map:

        agents = steps_to_use.agent_id_to_index

        # Loop through all agents and accumulate reward
        reward = 0
        for agent_id, agent_index in enumerate(agents):
            reward+= steps_to_use[agent_index].reward
            # print(f"{agent_id} reward: {steps_to_use[agent_index].reward}")

        # We want the group reward to be the mean over all agent rewards, so just divide the total reward by the number of agents.
        reward /= self.get_num_agents()



        return reward, terminal


    def env_done(self, terminal_steps) -> bool:
        if len(terminal_steps) > 0:
            return True
        else:
            return False



    # def discount_spec(self) -> Dict[str, specs.BoundedArray]:
    #     """Discount spec.
    #     Returns:
    #         Dict[str, specs.BoundedArray]: spec for discounts.
    #     """
    #     # print("Start execute discount_spec")
    #     discount_specs = {}
    #     for agent in self.all_agents_byname:
    #         discount_specs[agent] = specs.BoundedArray(
    #             (), np.float32, minimum=0.0, maximum=1.0
    #         )
    #     return discount_specs

    # def reward_spec(self) -> Dict[str, specs.Array]:
    #     """Reward spec.
    #     Returns:
    #         Dict[str, specs.Array]: spec for rewards.
    #     """
    #     # print("Start execute reward_spec")
    #     reward_specs = {}
    #     for agent in self.all_agents_byname:
    #         reward_specs[agent] = specs.Array((), dtype=np.float32)

    #     # print(f"Return reward spec:{reward_specs}")
    #     return reward_specs

    # def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
    #     """
    #     Action spec.

    #     Returns Dict[agent_id: np.ndarray] 
    #     """
    #     # print("Start execute action_spec")

    #     if self._action_space == "MultiDiscrete":
    #         nvec = np.zeros(self._behaviour_specs.action_spec.discrete_size)

    #         for i in range(self._behaviour_specs.action_spec.discrete_size):
    #             # This generates the nvec array, which contains the number of possible actions for each branch
    #             nvec[i] = self._behaviour_specs.action_spec.discrete_branches[i]

    #         maximum = nvec-1
    #         minimum = np.zeros(self._behaviour_specs.action_spec.discrete_size)

    #         # Will have to change the dtype to int, after changing MADQN for multidiscrete envs
    #         actions = specs.BoundedArray(
    #             shape= nvec.shape,
    #             dtype = np.int64,
    #             minimum=minimum,
    #             maximum=maximum,
    #             name=self._behaviour_name,
    #         )

    #     else:
    #         # print(f"Num values = {self._behaviour_specs.action_spec[1]}")
    #         actions = specs.DiscreteArray(
    #             num_values=self._behaviour_specs.action_spec[1][0],
    #             dtype = np.int64,

    #         )

    #     # ['agent_0', 'agent_1' ... etc]
    #     agents = self.all_agents_byname
        
    #     action_specs = {}
    #     for agent_id in agents:
    #         action_specs[agent_id] = actions

    #     return action_specs

    # def observation_spec(self) -> Dict[str, types.OLT]:
    #     """
    #     Only for visual observations at the moment
    #     If we use an encoder, the observation spec depends on the output shape of the encoder, not the actual visual observations
    #     """
    #     # print("Start execute obs_spec")
    #     observation_specs = {}
    #     agents = self.all_agents_byname #Max number of possible agents
    #     agents = dict.fromkeys(agents)

    #     # Gotta set the legal actions here, based on the length 3 vector observation which tells you whether or not the agent is looking at a portal/surface
    #     # Index 0 says whether or not the agent should be able to place portal 0, index 1 says the same for portal 1, and index 2 says whether the agent should
    #     # be able to attempt to remove a portal Therefore create a binary mask that looks likme [1 _ _ _] where the thre underscores are the contents of 
    #     # the obs[1] vector
        
    #     # legal_actions = self.action_spec()
    #     legal_actions = self.get_avail_actions()

    #     for agent in agents:
    #         obs = specs.BoundedArray(
    #             # self._behaviour_specs.observation_specs[0].shape instead of using agent as indicing because the agents have the same
    #             # observation spec
    #             shape = self._behaviour_specs.observation_specs[0].shape,
    #             # dtype = np.int64,
    #             dtype = np.float32,
    #             minimum= np.zeros(self._behaviour_specs.observation_specs[0].shape),
    #             maximum = np.ones(self._behaviour_specs.observation_specs[0].shape),
    #             name=self._behaviour_name
    #         )

    #         observation_specs[agent] = types.OLT(
    #             observation=obs,
    #             legal_actions=legal_actions[agent],
    #             terminal = specs.Array((1,), np.float32)
    #         )

    #     return observation_specs

    def get_avail_actions(self):
        # ds, ts = self._environment.get_steps(self._behaviour_name)

        # Check whether to use terminal or decision steps. Use decision steps for legal action
        # masking for all steps except if env is done, then use terminal steps
        # decision_steps, terminal_steps = self.get_steps()

        if self.env_done(self.terminal_steps):
            steps_to_use = self.terminal_steps
        else:
            steps_to_use = self.decision_steps

        legal_actions = []
        for i,agent in enumerate(self.all_agents_byname):
            # The first action should always be legal, as it is the action of "No action"
            legal_actions_list = [1]
            # obs[1] contains the length 2 vector that says whether those two actions are legal
            # NOTE: The below commented is deprecated because we use more obs[1] stuff as part of the global state, so we cant loop over everything for the legal mask
            # for action in steps_to_use[i].obs[1]:
            #     legal_actions_list.append(np.int32(action))

            for j in range(2):
                legal_actions_list.append(np.int32(steps_to_use[i].obs[1][j]))

            # The above legal action masking step only masks the first 3 actions, i.e
            # portal placement and no op. Branch 1 in the multidiscrete space.
            # Also add masking for all other actions, but make them always true
            if self._action_space == "Discrete":
                # self.action_mask contains a list of all 1's to mask all actions that are
                # always legal
                if not self._action_mask:
                    length = self.get_num_actions() - len(legal_actions_list)
                    self._action_mask = [1] * length
                fully_masked_actions = legal_actions_list + self._action_mask

            elif self._action_space == "MultiDiscrete":
                raise NotImplementedError


            legal_actions.append(np.array(fully_masked_actions))

        # print(f"The set of legal actions for this step is: {legal_actions}")
        return legal_actions


    # def extra_spec(self) -> Dict[str, specs.BoundedArray]:
    #     """
    #     extra_spec contains, for now, just the global state spec. 
        
    #     """

    #     # print("Start execute extra_spec")
    #     # This is the shape of one observation. I want two, concatenated side by side
    #     shape = self._behaviour_specs.observation_specs[0].shape
        
    #     # Generate an empty array with the shapoe of one observation, and then concatenate it to itself to be able to obtain the shape of
    #     # two concatenated observations.
    #     # Shape then contains the shape of two side-by-side observations, in this case (84, 168, 3). Tis is so wrtong tho. Make it (2,84,84,3) rather
    #     a = np.zeros(shape)
    #     # b = np.concatenate((a,a), axis = 1)
    #     b = np.stack((a,a), axis = 0)
    #     b = b.flatten()
    #     shape = b.shape

    #     if self._return_state_info:
    #         return {"s_t": specs.BoundedArray(
    #             shape = shape,
    #             dtype = np.float32,
    #             minimum = np.zeros(shape),
    #             maximum = np.ones(shape),
    #         )}
    #     else:
    #         return {}
    #         # Get each agent's observation, convert it to a single boundedarray

    #     # return {}

    # def savegif(self):
    #     if time.time() - self.starttime > 60:
    #         self.starttime = time.time()
            
    #         # agent0obs = self.convert_stacks_to_list(self.agent0obs)
    #         # agent1obs = self.convert_stacks_to_list(self.agent1obs)

    #         agent0obs = self.agent0obs
    #         agent1obs = self.agent1obs

    #         agent0obs_int = [np.uint8(x*256) for x in agent0obs]
    #         agent1obs_int = [np.uint8(x*256) for x in agent1obs]


    #         clip0 = ImageSequenceClip(agent0obs_int, fps = 5)
    #         clip1 = ImageSequenceClip(agent1obs_int, fps = 5)

    #         clip0.write_gif(f"./gifs/agent_0_segment_{self.a0}.gif", fps = 5)
    #         clip1.write_gif(f"./gifs/agent_1_segment_{self.a1}.gif", fps = 5)
    #         self.a0+=1
    #         self.a1+=1

    #         self.agent0obs.clear()
    #         self.agent1obs.clear()
    #         # del agent0obs
    #         # del agent1obs
    #         print("Wrote gifs for agents")

    def convert_stacks_to_list(self, list_of_stacked_obs_to_convert):
        # This only works for grayscale observations
        # Get the number of stacked observations:
        print(f"obs shape: {list_of_stacked_obs_to_convert[0].shape}")
        num_stacked_obs = list_of_stacked_obs_to_convert[0].shape[2] #eg 84,84,4 -> 4 stacked obs

        flattened = []
        for stacked_obs in list_of_stacked_obs_to_convert:
            for i in range(num_stacked_obs):
                sobs = stacked_obs[:,:,i]
                flattened.append(np.expand_dims(sobs, axis = -1))

        return flattened
    
    def get_obs_spec(self):
        """
        Get the shape of observations that the agents return
        """
        
        return self.obs_shape
    
    def get_num_actions(self):
        """
        Return the number of actions in the environment
        Cases:
            MultiDiscrete:
                NotImplementedYet
            Discrete:
                Returns a single value
        """
        if self._action_space == "MultiDiscrete":
            raise NotImplementedError
        
        else:
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
        # I could probably lose the methods and directly get the stuff here, but it's clearner
        # this way if I want to change eg the num_actions stuff later
        obs_shape = self.get_obs_spec()
        n_actions = self.get_num_actions()
        n_agents = self.get_num_agents()
        
        env_info = {}
        env_info["obs_shape"] = obs_shape
        env_info["n_actions"] = n_actions
        env_info["n_agents"] = n_agents
        env_info["episode_limit"] = self.episode_limit

        # Use only if extra state info is available from the environment
        if self._behaviour_specs.observation_specs[1].shape[-1]>2:
            # The -2 is because the first 2 values are not part of the global state, the +6 is beacuse each agent's position and rotation is represented by 6 flaots,
            # and we have two agents, and only the first ones' values are included by default
            
            state_size_to_add = (len(self.possible_agents)-1)*6 # value is 6

            env_info["state_shape"] = self._behaviour_specs.observation_specs[1].shape[0] - 2 + state_size_to_add

        assert type(env_info["obs_shape"]) == tuple, "Environment is not returning a tuple as observation"

        return env_info
    
    def get_env_info(self):
        env_info = {}
        if self.step_count == self.episode_limit:
            env_info["episode_limit"] = True
        # print(f"step count: {self.step_count}")
        # print(f"ep_limit: {self.episode_limit}")
        return env_info

    @property
    def possible_agents(self)->List:
        """All possible agents in env."""
        # print("Start execute possible_agents")
        decision_steps, _ = self._environment.get_steps(self._behaviour_name)
        agents = []
        # for i in range(len(decision_steps)):
        for i in decision_steps.agent_id:
            # agents.append(i)
            agents.append(f"agent_{i}")
        return agents
                