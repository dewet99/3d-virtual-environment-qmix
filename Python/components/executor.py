import ray
import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (EngineConfigurationChannel,)
from wrappers.UnityParallelEnvWrapper_Torch import UnityWrapper
from mlagents_envs.base_env import ActionTuple
import pdb
from collections import deque
import numpy as np
import gc
from functools import partial
from utils.unity_utils import get_worker_id
import torch
from controllers.custom_controller import CustomMAC
from models.ICMModel_2 import ICMModel
from components.replay_buffer import EpisodeBatch
from utils.utils import OneHot, RunningMeanStdTorch
import torch.nn.functional as F
import torch.nn as nn 
import time

import traceback
import datetime
import sys
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from models.NatureVisualEncoder import NatureVisualEncoder


@ray.remote(num_cpus = 1,num_gpus=0.0001, max_restarts=20)
class Executor(object):
    def __init__(self,config, worker_id):
        super().__init__()
        # Set config items
        self.time_scale = config["time_scale"]
        self.env_path = config["executable_path"]
        self.episode_limit = config["episode_limit"]
        self.config = config
        self.batch_size = config["batch_size_run"]


        # The executor processes should always run on CPU, to save GPU resources for training
        self.device = torch.device("cpu")
        self.worker_id = worker_id

        # Set class variables
        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(time_scale=self.time_scale)

        try:
            self.env.close()
            self.unity_env.close()
        except:
            print("No envs open")

        unity_env = UnityEnvironment(file_name=self.env_path, worker_id=worker_id, seed=np.int32(np.random.randint(0, 120)), side_channels=[config_channel])
        # unity_env = UnityEnvironment(file_name='./unity/envs/Discrete_NoCur/Discrete_NoCur.x86_64', worker_id=get_worker_id())
        unity_env.reset()

        self.env = UnityWrapper(unity_env, episode_limit=self.episode_limit, config = self.config)
        self.env.reset()

        self.get_env_info()
        self.setup()
        self.setup_logger()


    def collect_experience(self):
        # print(f"Executor {self.worker_id} starting to collect experience")
        self.reset()
        episode_start = time.time()
        try:
            # global_steps = ray.get(self.parameter_server.return_environment_steps.remote())
            global_steps = ray.get(self.parameter_server.get_worker_steps_by_id.remote(self.worker_id))
        except Exception as e:
            print(e)
        terminated = False
        episode_return = 0

        self.mac.init_hidden(batch_size=self.batch_size, hidden_state=None)
        raw_observations = 0

        reward_episode = []
        intrinsic_reward = None
        try:
            while not terminated:

                raw_observations = np.uint8(self.env._get_observations()*255)
                # print(self.batch["obs"])

                # state is determined from raw obs after feature extraction
                # normalise the obs before you save them to the replay buffer
                # if self.config["contains_state"]:
                state = self.env._get_global_state_variables()

                pre_transition_data = {
                    "state": state,
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }
                
                try:
                    if self.config["use_burnin"]:
                        # store the hidden state in the replay buffer so that we can use them during training to init hidden

                        h_s = self.mac.hidden_states.detach()

                        hidden_state = {"hidden_state": h_s.unsqueeze(0)}
                        pre_transition_data.update(hidden_state)
                except Exception as e:
                    traceback.print_exc()
                
                    
                    


                # else:
                #     pre_transition_data = {
                #         "avail_actions": self.env.get_avail_actions(),
                #         "obs": raw_observations
                #     }

                self.batch.update(pre_transition_data, ts=self.t)


                # Pass the entire batch of experiences up till now to the 
                # Receive the actions for each agent at this timestep in a batch of size 1
                # This will change depending on whether I'm using a feature extraction network or not
                with torch.no_grad():
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=global_steps, test_mode=False)
                    reward, terminated, env_info = self.env.step(actions[0])

                

                reward_episode.append(reward)


                episode_return += reward


                post_transition_data = {
                    "actions": actions,
                    # "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    # terminated above says whether the agent terminated because they reached the end
                    # of the episode
                }

                self.batch.update(post_transition_data, ts=self.t)

                reward_data = {
                    # "actions": actions,
                        "reward": [(reward,)],
                    # "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    # terminated above says whether the agent terminated because they reached the end
                    # of the episode
                    }
                
                self.batch.update(reward_data, ts=self.t)

                self.t += 1

            raw_observations = np.uint8(self.env._get_observations()*255)

            last_data = {
                    "state": self.env._get_global_state_variables(),
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }
            
            self.batch.update(last_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=global_steps, test_mode=False)

            self.batch.update({"actions": actions}, ts=self.t)

            
            # Parameter server keeps track of global steps and episodes
            # Add the number of steps in this executor's episode to the global count
            self.parameter_server.add_environment_steps.remote(self.t)
            
            # Increment global episode count
            self.parameter_server.increment_total_episode_count.remote()


            self.parameter_server.accumulate_stats.remote(sum(reward_episode), time.time() - episode_start, self.t)
            self.parameter_server.accumulate_worker_steps_by_id.remote(self.worker_id, self.t)


            # print(f"Executor {self.worker_id} took {time.time() - episode_start} seconds to complete an episode")

            return self.batch
        except Exception as e:
            traceback.print_exc()
    

    def run(self):
        try:
            time.sleep(3)
            pupdates = 0
            while pupdates<self.config["t_max"]+5:

                # Sample every 5 parameter updates:
                pupdates = ray.get(self.parameter_server.get_parameter_update_steps.remote())
                if pupdates % self.config["worker_parameter_sync_frequency"] == 0:
                    # These two can be the same function but I leave as is for now
                    self.sync_with_parameter_server()
                    self.sync_with_param_server_encoder()


                # if pupdates % self.config["log_every"] == 0 and pupdates>0 and self.config["log_histograms"]:
                #     for key, value in self.mac.named_parameters():
                #         self.histograms_writer.add_histogram(f"Executor_Histograms/{key}", value, ray.get(self.parameter_server.return_environment_steps.remote()))


                #     # print(f"Worker {self.worker_id} is synced with parameter server")

                # print(f" Worker {self.worker_id} will now start to collect one experience")

                # sync the executor's encoder parameters with the parameter server's encoder paramts:
                # self.sync_with_param_server_encoder()

                
                
                episode_batch = self.collect_experience()

                if self.config["save_obs_for_debug"]:
                    np.save("epbatch_obs", episode_batch["obs"])
                    self.config["save_obs_for_debug"] = False

                # print(f" Worker {self.worker_id} collected one experience")

                # Trying to use shared memory
                # episode_batch_reference = ray.put(episode_batch)
                # print("Attempt episode insert")
                # blocking call, task has to complete before I can continue
                # ray.get(self.remote_buffer.insert_episode_batch.remote(episode_batch))
                self.remote_buffer.insert_episode_batch.remote(episode_batch)

                # print("Episode intertsed")
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()


    def set_remote_objects(self, remote_buffer, parameter_server):
        self.remote_buffer = remote_buffer
        self.parameter_server = parameter_server


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.icm_reward = 0
        self.li = 0
        self.lf = 0
        self.L_I = 0
        self.L_F = 0
    
    def setup(self):
        scheme, groups, preprocess = self.generate_scheme()
        
        self.encoder = NatureVisualEncoder(self.config["obs_shape"][0],
                                           self.config["obs_shape"][1],
                                           self.config["obs_shape"][2],
                                           self.config,
                                           device = self.device
                                           )
        # .cuda()
        
        if self.config["load_pretrained_model"]:
            self.encoder.load_state_dict(torch.load("./encoder_weights/encoder_NEW.pth"))

        
        self.mac = CustomMAC(self.config, encoder = self.encoder, device = self.device)

        # if self.config["use_transfer"]:
        #     self.encoder.load_state_dict(torch.load(self.config["models_2_transfer_path"] + "/encoder.th"))
        #     self.mac.load_models(self.config["models_2_transfer_path"])

        #     self.encoder.to(self.device)
        #     self.mac.agent.to(self.device)

        # if not self.config["use_per"]:
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.config["batch_size_run"], self.config["episode_limit"]+1, preprocess = preprocess, device = "cpu")
        # else:
        #     self.new_batch = partial(PER_EpisodeBatch, scheme, groups, self.config["batch_size_run"], self.config["episode_limit"]+1, preprocess = preprocess, device = "cpu")

    def get_env_info(self):
        self.config["obs_shape"] = self.env.obs_shape
        self.env_info = self.env.get_init_env_info()
        self.config["n_actions"] = self.env_info["n_actions"]

    def setup_logger(self):
        # print(self.config["load_models_from"])
        self.log_dir = "results/" + datetime.datetime.now().strftime("%d_%m_%H_%M")
        self.histograms_writer = SummaryWriter(log_dir= self.log_dir + "/histograms")

    def close_env(self):
        self.env.close()

    def generate_scheme(self):
        self.config["state_shape"] = self.env_info["state_shape"]

        scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents", "dtype": torch.uint8},
            "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
            "avail_actions": {"vshape": (self.env_info["n_actions"],), "group": "agents", "dtype": torch.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            }
        
        # if self.config["curiosity"]:
        #     icm_reward = {"icm_reward": {"vshape": (1,)},}
        #     scheme.update(icm_reward)

        if self.config["use_burnin"]:
            hidden_states = {"hidden_state": {"vshape": (1, 2,self.config["rnn_hidden_dim"]), "dtype": torch.float32}}
            scheme.update(hidden_states)

        # if self.config["useNoisy"]:
        #     raise NotImplementedError
        
        groups = {
        "agents": self.config["num_agents"]
        }

        preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=self.config["n_actions"])])
        }

        return scheme, groups, preprocess
    
    def retrieve_updated_config(self):
        return self.config
    
    def sync_with_parameter_server(self):
        # receive the stored parameters from the server using ray.get()

        new_params = ray.get(self.parameter_server.return_params.remote())
        # print(f"New params typee: {type(new_params)}")
        # print(new_params)

        for param_name, param_val in self.mac.named_parameters():
            if param_name in new_params:
                param_data = torch.tensor(ray.get(new_params[param_name])).to(self.device)
                param_val.data.copy_(param_data)
        
        # copy the received neural network weights to its own



    def sync_with_param_server_encoder(self):
        new_params = ray.get(self.parameter_server.return_encoder_params.remote())
        # print(f"New params typee: {type(new_params)}")
        # print(new_params)

        for param_name, param_val in self.encoder.named_parameters():
            if param_name in new_params:
                param_data = torch.tensor(ray.get(new_params[param_name])).to(self.device)
                param_val.data.copy_(param_data)

    def sync_with_param_server_ICM_encoder(self):
        new_params = ray.get(self.parameter_server.return_ICM_encoder_params.remote())
        # print(f"New params typee: {type(new_params)}")
        # print(new_params)

        for param_name, param_val in self.icm.icm_encoder.named_parameters():
            if param_name in new_params:
                param_data = torch.tensor(ray.get(new_params[param_name])).to(self.device)
                param_val.data.copy_(param_data)





