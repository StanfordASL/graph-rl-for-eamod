"""
SAC-GNN
-------
This file contains the SAC-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks
(4) SAC:
    Soft actor critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from src.algos.pax_flows_solver import PaxFlowsSolver
from src.algos.reb_flows_solver import RebalFlowSolver
from src.misc.utils import dictsum
import random
import time


class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    Replay buffer for SAC agents
    """
    MAX_SIZE = 1000000  # Max size of the replay buffer

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        # If the buffer size exceeds the max size, remove the oldest data
        if len(self.data_list) >= ReplayData.MAX_SIZE:
            self.data_list.pop(0)
            self.rewards.pop(0)
        
        self.data_list.append(PairData(data1.edge_index, data1.x, torch.as_tensor(
            reward), torch.as_tensor(action), data2.edge_index, data2.x))
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=['x_s', 'x_t'])
            batch.reward = (batch.reward-mean)/(std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


#########################################
############## ACTOR ####################
#########################################

class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, city='NY'):
        super().__init__()
        self.city = city
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        
        if concentration.isnan().any():
            raise ValueError('Nan in concentration')
        
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)

        return action, log_prob

#########################################
############## CRITICS ##################
#########################################


class GNNCritic1(nn.Module):
    """
    Architecture 1, GNN, Pointwise Multiplication, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action*10
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x*action  # pointwise multiplication (B,N,21)
        x = x.sum(dim=1)  # (B,21)
        x = F.relu(self.lin1(x))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic2(nn.Module):
    '''
    Architecture 2, GNN, Readout, Concatenation, FC
    '''

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels+act_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, 21)  # (B,N,21)
        x = torch.sum(x, dim=1)  # (B, 21)
        concat = torch.cat([x, action], dim=-1)  # (B, 21+N)
        x = F.relu(self.lin1(concat))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # B
        return x


class GNNCritic3(nn.Module):
    """
    Architecture 3: Concatenation, GNN, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(22, 22)
        self.lin1 = nn.Linear(22, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        cat = torch.cat([state, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        out = F.relu(self.conv1(cat, edge_index))
        x = out + cat
        x = x.reshape(-1, self.act_dim, 22)  # (B,N,22)
        x = F.relu(self.lin1(x))  # (B, H)
        x = F.relu(self.lin2(x))  # (B, H)
        x = torch.sum(x, dim=1)  # (B, 22)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic4(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout 
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic5(nn.Module):
    """
    Architecture 5, GNN, Pointwise Multiplication, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action+1
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x*action  # pointwise multiplication (B,N,21)
        x = F.relu(self.lin1(x))  # (B,N,H)
        x = F.relu(self.lin2(x))  # (B,N,H)
        x = x.sum(dim=1)  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x

#########################################
############## SAC AGENT ################
#########################################


class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,  # environment
        input_size,  # state dimnension
        hidden_size=32,  # hidden units in MLP
        alpha=0.2,  # entropy coefficient
        gamma=0.99,  # discount factor
        polyak=0.995,  # polyak averaging for Q-targets
        p_lr=1e-4,  # actor learning rate
        q_lr=5e-4,  # critic learning reate
        device=torch.device("cpu"),
        use_automatic_entropy_tuning=False,
        clip=5,
        critic_version=4,
        city='NY'
    ):
        super(SAC, self).__init__()
        self.env = env
        self.city = city
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_dim = env.number_nodes

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.clip = clip
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.nodes = env.number_nodes

        self.replay_buffer = ReplayData(device=device)
        # nnets
        self.actor = GNNActor(
            self.input_size, self.hidden_size, act_dim=self.act_dim, city=self.city)
        self.actor.to(self.device)
        print(self.device)
        print(self.actor)

        if critic_version == 1:
            GNNCritic = GNNCritic1
        if critic_version == 2:
            GNNCritic = GNNCritic2
        if critic_version == 3:
            GNNCritic = GNNCritic3
        if critic_version == 4:
            GNNCritic = GNNCritic4
        if critic_version == 5:
            GNNCritic = GNNCritic5

        self.critic1 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic1.to(self.device)
        
        self.critic2 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2.to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        print(self.critic1)

        self.critic1_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic1_target.to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2_target.to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            a, _ = self.actor(x, edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def compute_loss_q(self, data):
        (state_batch,
         edge_index,
         next_state_batch,
         edge_index2,
         reward_batch,
         action_batch) = (data.x_s, data.edge_index_s, data.x_t, data.edge_index_t, data.reward, data.action.reshape(-1, self.nodes))
        
        state_batch = state_batch.to(self.device)
        edge_index = edge_index.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        edge_index2 = edge_index2.to(self.device)
        reward_batch = reward_batch.to(self.device)
        action_batch = action_batch.to(self.device)

        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from current policy
            
            try:
                a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            except ValueError:
                raise ValueError("ValueError")
            
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = reward_batch + self.gamma * \
                (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1.float(), backup.float())
        loss_q2 = F.mse_loss(q2.float(), backup.float())

        return loss_q1.float(), loss_q2.float()

    def compute_loss_pi(self, data):

        state_batch, edge_index = data.x_s, data.edge_index_s,

        try:
            actions, logp_a = self.actor(state_batch, edge_index)
        except ValueError:
            raise ValueError("ValueError")
        
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index,  actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()

        return loss_pi

    def update(self, data):

        try:
            loss_q1, loss_q2 = self.compute_loss_q(data)
        except ValueError:
            raise ValueError("ValueError")

        loss_q1 = loss_q1.float()
        loss_q2 = loss_q2.float()

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(),  self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(),  self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        
        try:
            loss_pi = self.compute_loss_pi(data)
        except ValueError:
            raise ValueError("ValueError")
        
        loss_pi.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(
            actor_params, lr=self.p_lr
        )
        optimizers["c1_optimizer"] = torch.optim.Adam(
            critic1_params, lr=self.q_lr
        )
        optimizers["c2_optimizer"] = torch.optim.Adam(
            critic2_params, lr=self.q_lr
        )

        return optimizers
    

    ########

    def test_agent(self, test_episodes, env, pax_flows_solver, rebal_flow_solver, parser):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rl_inference_time = []
        for i_episode in epochs:
            desired_accumulations_spatial_nodes = np.zeros(env.scenario.spatial_nodes)
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_time = []

            obs = env.reset(bool_sample_demand=True, seed=i_episode)

            actions = []
            done = False

            while (not done):         
                time_start = time.time()       
                pax_flows_solver.update_constraints()
                pax_flows_solver.update_objective()
                obs, paxreward, done, info_pax = env.pax_step(pax_flows_solver=pax_flows_solver, episode=i_episode)
                eps_reward += paxreward

                o = parser.parse_obs(obs)

                action_rl = self.select_action(o, deterministic=True)
                actions.append(action_rl)

                total_idle_acc = sum(env.acc[n][env.time+1] for n in env.nodes)
                desired_acc = {env.nodes[i]: int(action_rl[i] *total_idle_acc) for i in range(env.number_nodes)} # over nodes
                total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
                missing_cars = total_idle_acc - total_desiredAcc
                most_likely_node = np.argmax(action_rl)
                if missing_cars != 0:
                    desired_acc[env.nodes[most_likely_node]] += missing_cars   
                    total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
                assert abs(total_desiredAcc - total_idle_acc) < 1e-5
                for n in env.nodes:
                    assert desired_acc[n] >= 0
                for n in env.nodes:
                    desired_accumulations_spatial_nodes[n[0]] += desired_acc[n]
                
                rebal_flow_solver.update_constraints(desired_acc, env)
                rebal_flow_solver.update_objective(env)
                rebAction = rebal_flow_solver.optimize()

                # take action in environment
                new_obs, rebreward, rebreward_internal, done, info_reb = env.reb_step(rebAction)
                eps_reward += rebreward
                eps_served_demand += info_pax["served_demand"]
                eps_rebalancing_cost += info_reb["rebalancing_cost"]
                eps_time.append(time.time() - time_start)
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_rl_inference_time.append(np.mean(eps_time))

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
            np.mean(episode_rl_inference_time),
        )

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint["model"])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
