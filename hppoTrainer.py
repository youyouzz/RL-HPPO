import argparse
import os

import numpy as np
import torch

from hppo.hppo_actionmask import *
from hppo.hppo_utils import *
from util.configuration import *
from util.util import *
from env.wet_rl_env import WetEnv
from env.data_loading import Data

@timer
class Trainer(object):
    """
    A RL trainer.
    """

    def __init__(self, args):
        self.version_no = args.wetConfig['version_no']
        self.mode = args.wetConfig["mode"]
        self.data_source = args.wetConfig["data_source"]
        self.experiment_name = args.experiment_name
        
        self.device = args.device
        self.max_episodes = args.max_episodes
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.agent_save_freq = args.agent_save_freq
        self.agent_update_freq = args.agent_update_freq

        # agent's hyperparameters
        self.mid_dim = args.mid_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_actor_param
        self.lr_std = args.lr_std
        self.lr_decay_rate = args.lr_decay_rate
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam
        self.epochs_update = args.epochs_update
        self.v_iters = args.v_iters
        self.eps_clip = args.eps_clip
        self.max_norm_grad = args.max_norm_grad
        self.init_log_std = args.init_log_std
        self.coeff_dist_entropy = args.coeff_dist_entropy
        self.random_seed = args.random_seed
        self.if_use_active_selection = args.if_use_active_selection


        # For save
        self.save_path = 'log/' + self.version_no + '_' + self.experiment_name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.record_mark = args.record_mark
        # self.policy_save = os.path.join(self.file_to_save, 'policy/{}/{}'.format(self.record_mark))
        # self.results_save = os.path.join(self.file_to_save, 'results/{}/{}'.format(self.record_mark))
        # os.makedirs(self.policy_save, exist_ok=True)
        # os.makedirs(self.results_save, exist_ok=True)

        
        self.data = Data(version_no=self.version_no, mode=self.mode, data_source=self.data_source)
        self.env = WetEnv(self.data)
        self.machine_qty = self.env.machine_qty # 7
        self.obs_dim = self.env.observation_space.shape[0] # 24
        self.action_dis_dim = self.env.action_space.shape[0] # 7
        self.action_len = self.env.action_len # 3
        
        
        self.history = {}
        self.monitor = Monitor(self.data)
        
        

    def push_history_dis(self, obs, action_mask, act_dis, logp_act_dis, val):
        self.history = {
            'obs': obs,
            'action_mask': action_mask,
            'act_dis': act_dis,
            'logp_act_dis': logp_act_dis,
            'val': val
        }
        
    def push_history_hybrid(self, obs, action_mask, act_dis, act_con, logp_act_dis, logp_act_con, val):
        self.history = {
            'obs': obs,
            'action_mask': action_mask,
            'act_dis': act_dis,
            'act_con': act_con,
            'logp_act_dis': logp_act_dis,
            'logp_act_con': logp_act_con,
            'val': val
        }


    def unbatchify(self, value_action_logp: dict):
        state_value = value_action_logp[0]
        actions = value_action_logp[1]
        logp_actions = value_action_logp[2]
        
        # actions = np.array([action_dis, action_con])
        # logp_actions = np.array([log_prob_dis, log_prob_con])
        
        return state_value, actions, logp_actions


    def initialize_agents(self, random_seed):
        """
        Initialize environment and agent.

        :param random_seed: could be regarded as worker index
        :return: instance of agent and env
        """
        
        # return PPO_Hybrid(self.obs_dim, self.action_dis_dim, self.action_len, self.action_con_dim, self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update,self.v_iters, self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device, self.lr_std, self.init_log_std, self.if_use_active_selection)
        return PPO_Discrete(self.obs_dim, self.action_dis_dim, self.action_len, self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update, self.v_iters, self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device)
            
        

    def train(self, worker_idx):
        """

        :param worker_idx:
        :return:
        """
        

        agent = self.initialize_agents(worker_idx)

        norm_mean = np.zeros(self.obs_dim)
        norm_std = np.ones(self.obs_dim)

        i_episode = 0
        

        ### TRAINING LOGIC ###
        while i_episode < self.max_episodes:
            # collect an episode
            with torch.no_grad():
                state, info = self.env.reset()
                action_mask = self.env.action_mask
                next_state = state
                total_reward = 0

                while True:
                    # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                    # mean and std retrieve from the last update's buf, in other word observations normalization
                    observations_norm = (state - norm_mean) / np.maximum(norm_std, 1e-6)
                    # Select action with policy
                    value_action_logp = agent.select_action(observations_norm, action_mask)
                    values, actions, logp_actions = self.unbatchify(value_action_logp)

                    next_state, reward, done, truncated, info = self.env.step(actions)

                    self.push_history_dis(state, action_mask, actions, logp_actions, values)
                    agent.buffer.store_dis(self.history['obs'], self.history['action_mask'], self.history['act_dis'], reward, self.history['val'], self.history['logp_act_dis'])
                    
                    total_reward += reward

                    state = next_state
                    action_mask = self.env.action_mask

                    if done:
                        if i_episode % 100 == 0:
                            print("record_eqp_plan:\n", self.env.record_eqp_plan)
                            print("record_wip_move:\n", self.env.record_wip_move)
                            print("total move qty =", sum(sum(self.env.record_wip_move)))
                            print("record_acid_density:\n", self.env.record_acid_density)
                            print("record_acid_lifetime:\n", self.env.record_acid_lifetime)
                        i_episode += 1
                        agent.buffer.finish_path(0)
                        break
                print(f"Episode {i_episode} - Total Reward: {total_reward}")
                self.monitor.push_history('total_reward', total_reward)
                
                
                
                # logger.info(f"Episode {i_episode} - Total Reward: {total_reward}")

            if i_episode % self.agent_update_freq == 0:
                norm_mean = agent.buffer.filter()[0]
                norm_std = agent.buffer.filter()[1]
                if i_episode > self.agent_save_freq:
                    agent.update(self.batch_size)
                agent.buffer.clear()

    def save_data(self):
        total_reward_history = self.monitor.get_history('total_reward')
        # save the total reward history
        np.save(self.save_path + 'total_reward_history.npy', total_reward_history)
        # plot the total reward history
        self.monitor.plot('total_reward', self.save_path)
        
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_episodes', type=int, default=1001, help='The max episodes per agent per run.')
    parser.add_argument('--buffer_size', type=int, default=6000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=64, help='The sample batch size.')
    parser.add_argument('--agent_save_freq', type=int, default=10, help='The frequency of the agent saving.')
    parser.add_argument('--agent_update_freq', type=int, default=10, help='The frequency of the agent updating.')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='The learning rate of actor_con.')   # carefully!
    parser.add_argument('--lr_actor_param', type=float, default=0.001, help='The learning rate of critic.')
    parser.add_argument('--lr_std', type=float, default=0.004, help='The learning rate of log_std.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.995, help='Factor of learning rate decay.')
    parser.add_argument('--mid_dim', type=list, default=[256, 128, 64], help='The middle dimensions of both nets.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted of future rewards.')
    parser.add_argument('--lam', type=float, default=0.8,
                        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument('--epochs_update', type=int, default=20,
                        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--v_iters', type=int, default=1,
                        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument('--target_kl_dis', type=float, default=0.025,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--target_kl_con', type=float, default=0.05,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='The clip ratio when calculate surr.')
    parser.add_argument('--max_norm_grad', type=float, default=5.0, help='max norm of the gradients.')
    parser.add_argument('--init_log_std', type=float, default=-1.0,
                        help='The initial log_std of Normal in continuous pattern.')
    parser.add_argument('--coeff_dist_entropy', type=float, default=0.005,
                        help='The coefficient of distribution entropy.')
    parser.add_argument('--random_seed', type=int, default=1, help='The random seed.')
    parser.add_argument('--record_mark', type=str, default='renaissance',
                        help='The mark that differentiates different experiments.')
    parser.add_argument('--if_use_active_selection', type=bool, default=False,
                        help='Whether use active selection in the exploration.')
    parser.add_argument('--experiment_name', type=str, default='dis', help='The name of the experiment.')

    version_no = "RTS-T2-20240507164500"
    mode = "main_train"
    data_source = "pk"
    wetConfig = {"version_no": version_no, "mode": mode, "data_source": data_source}
    parser.add_argument('--wetConfig', type=dict, default=wetConfig, help='wet config')
    
    args = parser.parse_args()


    # training through multiprocess
    trainer = Trainer(args)
    trainer.train(1)
    trainer.save_data()
