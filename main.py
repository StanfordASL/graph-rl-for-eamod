from __future__ import print_function
import argparse
import os
import gurobipy as gp
from tqdm import trange
import numpy as np
from src.algos.pax_flows_solver import PaxFlowsSolver
from src.algos.reb_flows_solver import RebalFlowSolver
import torch
import json
import os
import pickle
import time
import copy

from src.envs.amod_env import Scenario, AMoD
from src.algos.sac import SAC
from src.misc.utils import dictsum

from torch_geometric.data import Data

class GNNParser():
    """
    Parser converting raw environment observations to agent input.
    """

    def __init__(self, env, T=10, scale_factor=0.01, scale_price=0.1, input_size=20):
        super().__init__()
        self.env = env
        self.T = T
        self.scale_factor = scale_factor
        self.price_scale_factor = scale_price
        self.input_size = input_size

    def parse_obs(self, obs):
        # nodes
        x = torch.cat((
            torch.tensor([float(n[1])/self.env.scenario.number_charge_levels for n in self.env.nodes]).view(1, 1, self.env.number_nodes).float(),
            torch.tensor([obs[0][n][self.env.time+1] * self.scale_factor for n in self.env.nodes]).view(1, 1, self.env.number_nodes).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.scale_factor for n in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes).float(),
            torch.tensor([[sum([self.env.price[o[0], j][t]*self.scale_factor*self.price_scale_factor*(self.env.demand[o[0], j][t])*((o[1]-self.env.scenario.energy_distance[o[0], j]) >= int(not self.env.scenario.charging_stations[j]))
                          for j in self.env.region]) for o in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes).float()),
                      dim=1).squeeze(0).view(2+self.T + self.T, self.env.number_nodes).T
        # edges
        edges = []
        for o in self.env.nodes:
            for d in self.env.nodes:
                if (o[0] == d[0] and o[1] == d[1]):
                    edges.append([o, d])
        edge_idx = torch.tensor([[], []], dtype=torch.long)
        for e in edges:
            origin_node_idx = self.env.nodes.index(e[0])
            destination_node_idx = self.env.nodes.index(e[1])
            new_edge = torch.tensor([[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx = torch.cat((edge_idx, new_edge), 1)
        edge_index = torch.cat((edge_idx, self.env.gcn_edge_idx), 1)

        data = Data(x, edge_index)
        return data


def create_scenario(json_file_path, energy_file_path, seed=10):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    spatial_nodes = data['spatialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_levels_per_charge_step = data['chargeLevelsPerChargeStep']
    chargers = data['chargeLocations']
    cars_per_station_capacity = data['carsPerStationCapacity']
    p_energy = data["energy_prices"]
    time_granularity = data["timeGranularity"]
    operational_cost_per_timestep = data['operationalCostPerTimestep']

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, cars_per_station_capacity = cars_per_station_capacity, number_charge_levels=number_charge_levels, charge_levels_per_charge_step=charge_levels_per_charge_step, 
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc, p_energy=p_energy, time_granularity=time_granularity, operational_cost_per_timestep=operational_cost_per_timestep)
    return scenario

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--spatial_nodes', type=int, default=5, metavar='N',
                    help='number of spatial nodes (default: 5)')
parser.add_argument('--city', type=str, default='NY', metavar='N',
                    help='city (default: NY)')
parser.add_argument('--zeroShotCity', type=bool, default=False,
                    help='whether to try different city')
parser.add_argument('--zeroShotNodes', type=int, default=0, 
                    help='num nodes in model to load')
parser.add_argument('--scratch', type=bool, default=False,
                    help='whether to start training from scratch')
parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--equal_distr_baseline', type=bool, default=False,
                    help='activates the equal distribution baseline.')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=9000, metavar='N',
                    help='number of episodes to train agent (default: 9k)')
parser.add_argument('--T', type=int, default=84, metavar='N',
                    help='Time horizon for the A2C')
parser.add_argument('--lr_a', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the actor')
parser.add_argument('--lr_c', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the critic')
parser.add_argument('--grad_norm_clip_a', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the actor')
parser.add_argument('--grad_norm_clip_c', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the critic')

parser.add_argument("--batch_size", type=int, default=256,
                    help='defines batch size')
parser.add_argument("--alpha", type=float, default=0.3,
                    help='defines entropy coefficient')
parser.add_argument("--hidden_size", type=int, default=256,
                    help='defines hidden units in the MLP layers')
parser.add_argument("--rew_scale", type=float, default=0.01,
                    help='defines reward scale')
parser.add_argument("--critic_version", type=int, default=4,
                    help='defines the critic version (default: 4)')

# required arg
parser.add_argument("--run_id", type=int, required=True,
                    help='defined unique ID for run')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
lr_a = args.lr_a
lr_c = args.lr_c
grad_norm_clip_a = args.grad_norm_clip_a
grad_norm_clip_c = args.grad_norm_clip_c
use_equal_distr_baseline = args.equal_distr_baseline
seed = args.seed
test = args.test
T = args.T
num_sn = args.spatial_nodes
city = args.city

zeroShotCity = args.zeroShotCity
zeroShotNodes = args.zeroShotNodes

# problem_folder = 'NY/ClusterDataset1'
# file_path = os.path.join('data', problem_folder,  'd1.json')
# problem_folder = 'NY_5'
# file_path = os.path.join('data', problem_folder,  'NY_5.json')
# problem_folder = 'SF_5_clustered'
# file_path = os.path.join('data', problem_folder,  'SF_5_short.json')

if city == 'NY':
    problem_folder = 'NY'
    file_path = os.path.join('data', problem_folder, str(num_sn), f'NYC_{num_sn}.json')
else:
    problem_folder = 'SF'
    file_path = os.path.join('data', problem_folder, str(num_sn), f'SF_{num_sn}.json')

experiment = 'training_' + file_path + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + '_new' + str(args.run_id)
energy_dist_path = os.path.join('data', problem_folder, str(num_sn), 'energy_distance.npy')

scenario = create_scenario(file_path, energy_dist_path)
env = AMoD(scenario)

print("Number of edges: " + str(len(env.scenario.edges)))
print("Number of spatial nodes: " + str(len(env.scenario.G_spatial.nodes)))
print("Number of nodes: " + str(len(env.scenario.G.nodes)))

# Initialize A2C-GNN
# NY
if city == 'NY':
    scale_factor = 0.01
    scale_price = 0.1
# SF
else:
    scale_factor = 0.00001
    scale_price = 0.1
# NY 5 
et = experiment

# if zeroShotCity or zeroShotNodes, temporarily alter experiment name accordingly
if zeroShotCity:
    experiment += '_zeroShotCity'
if zeroShotNodes:
    experiment += '_zeroShotNodes'

experiment = et

if city == 'NY':
    checkpoint_path = f"NYC_{num_sn}_{args.max_episodes}_{args.T}_{args.run_id}"
else:
    checkpoint_path = f"SF_{num_sn}_{args.max_episodes}_{args.T}_{args.run_id}"

parser = GNNParser(env, T=T, scale_factor=scale_factor, scale_price=scale_price)
run_id = args.run_id

model = SAC(
    env=env,
    input_size=(2*T + 2),
    hidden_size=args.hidden_size,
    alpha=args.alpha,
    use_automatic_entropy_tuning=False,
    critic_version=args.critic_version,
    device=device,
    city=city
).to(device);

train_episodes = args.max_episodes  # set max number of training episodes
epochs = trange(train_episodes)  # epoch iterator
if args.test:
    epochs = trange(1)

best_reward = -np.inf  # set best reward
best_reward_test = -np.inf  # set best reward

if zeroShotCity or (zeroShotNodes > 0):
    if zeroShotCity:
        if city == 'NY':
            checkpoint_path = f'SF_{num_sn}_{train_episodes}_{args.T}_{run_id}'
            # scale_factor = 0.00001
            # scale_price = 0.1
        else:
            checkpoint_path = f'NYC_{num_sn}_{train_episodes}_{args.T}_{run_id}'
            scale_factor = 0.01
            scale_price = 0.1
    else:
        if city == 'NY':
            checkpoint_path = f'NYC_{zeroShotNodes}_{train_episodes}_{args.T}_{run_id}'
        else:
            checkpoint_path = f'SF_{zeroShotNodes}_{train_episodes}_{args.T}_{run_id}'
    model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth')
    epochs = trange(1)
else:
    if not test:
        model.train()  # set model in train mode
    
        if not args.scratch:
            if not args.resume:
                warm_start_num_sn = num_sn - 5
                if city == 'NY':
                    checkpoint_path = f'NYC_{warm_start_num_sn}_{train_episodes}_{args.T}_{run_id}'
                else:
                    checkpoint_path = f'SF_{warm_start_num_sn}_{train_episodes}_{args.T}_{run_id}'
                model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth');
            else:
                model.load_checkpoint(path=f'checkpoint/{checkpoint_path}.pth');
    else:
        model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth');

total_demand_per_spatial_node = np.zeros(env.number_nodes_spatial)
for region in env.nodes_spatial:
    for destination in env.nodes_spatial:
        for t in range(env.tf):
            total_demand_per_spatial_node[region] += env.demand[region,destination][t]

for i_episode in epochs:
    desired_accumulations_spatial_nodes = np.zeros(env.scenario.spatial_nodes)
    obs = env.reset(bool_sample_demand=True, seed=i_episode) #initialize environment
    
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
    episode_times = []
    actions = []

    current_eps = []
    done = False
    step = 0
    time_a_end = 0
    time_b_end = 0
    time_c_end = 0
    time_d_end = 0
    time_e_end = 0

    while (not done):
        time_i_start = time.time()
        if step > 0:
            obs1 = copy.deepcopy(o)
        
        time_2 = time.time()
        # take matching step (Step 1 in paper)
        if step == 0 and i_episode == 0:
            # initialize optimization problem in the first step
            pax_flows_solver = PaxFlowsSolver(env=env, gurobi_env=None)
        else:
            time_a = time.time()
            pax_flows_solver.update_constraints()
            time_a_end = time.time() - time_a

            time_b = time.time()
            pax_flows_solver.update_objective()
            time_b_end = time.time() - time_b
        time_2_end = time.time() - time_2
        
        time_3 = time.time()
        obs, paxreward, done, info_pax = env.pax_step(pax_flows_solver=pax_flows_solver, episode=i_episode)
        time_3_end = time.time() - time_3

        time_4 = time.time()
        o = parser.parse_obs(obs)
        time_4_end = time.time() - time_4

        episode_reward += paxreward
        if step > 0:
            rl_reward = (paxreward + rebreward)
            time_5 = time.time()
            model.replay_buffer.store(obs1, action_rl, args.rew_scale * rl_reward, o)
            time_5_end = time.time() - time_5

        time_6 = time.time()
        # sample from Dirichlet (Step 2 in paper)
        if test:
            try:
                action_rl = model.select_action(o, deterministic=True)
            except ValueError:
                model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth')
                action_rl = model.select_action(o, deterministic=True)
        else:
            try:
                action_rl = model.select_action(o)
            except ValueError:
                model.load_checkpoint(path=f'checkpoint/{checkpoint_path}.pth')
                action_rl = model.select_action(o)
        time_6_end = time.time() - time_6

        time_7 = time.time()
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
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
        time_7_end = time.time() - time_7

        time_8 = time.time()
        # solve minimum rebalancing distance problem (Step 3 in paper)
        if step == 0 and i_episode == 0:
        # initialize optimization problem in the first step
            rebal_flow_solver = RebalFlowSolver(env=env, desiredAcc=desired_acc, gurobi_env=None)
        else:
            time_c = time.time()
            rebal_flow_solver.update_constraints(desired_acc, env)
            time_c_end = time.time() - time_c
            time_d = time.time()
            rebal_flow_solver.update_objective(env)
            time_d_end = time.time() - time_d
        time_e = time.time()
        rebAction = rebal_flow_solver.optimize()
        time_e_end = time.time() - time_e
        time_8_end = time.time() - time_8

        time_9 = time.time()
        # Take action in environment
        new_obs, rebreward, rebreward_internal, done, info_reb = env.reb_step(rebAction)
        episode_reward += rebreward
        time_9_end = time.time() - time_9
        
        # track performance over episode
        episode_served_demand += info_pax['served_demand']
        episode_rebalancing_cost += info_reb['rebalancing_cost']
        episode_times.append(time.time() - time_i_start)

        # stop episode if terminating conditions are met
        step += 1
        if args.resume and i_episode > 100:
            if i_episode > 10:
                if (city == "SF") and not args.scratch:
                    for step in range(100):
                        batch = model.replay_buffer.sample_batch(
                            args.batch_size)  # sample from replay buffer
                        model = model.float()
                        try:
                            model.update(data=batch)  # update model
                        except ValueError:
                            model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth')
                else:
                    for step in range(50):
                        batch = model.replay_buffer.sample_batch(
                            args.batch_size)  # sample from replay buffer
                        model = model.float()
                        try:
                            model.update(data=batch)  # update model
                        except ValueError:
                            model.load_checkpoint(path=f'checkpoint/{checkpoint_path}_test.pth')
        else:
            continue

    # Log performance
    epochs.set_description(
        f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f} | Avg. Time: {np.array(episode_times).mean():.2f}sec")
    print(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f} | Avg. Time: {np.array(episode_times).mean():.2f}sec")
    
    # Checkpoint best performing model
    if episode_reward >= best_reward:
        path = os.path.join('checkpoint', f'{checkpoint_path}.pth')
        if not test:
            model.save_checkpoint(path=path)
        best_reward = episode_reward
        best_rebal_cost = episode_rebalancing_cost
        best_served_demand  = episode_served_demand
        best_model = model

    if i_episode % 10 == 0:  # test model every 10th episode
        test_reward, test_served_demand, test_rebalancing_cost, test_time = model.test_agent(
            1, env, pax_flows_solver, rebal_flow_solver, parser=parser)
        if test_reward >= best_reward_test:
            best_reward_test = test_reward
            path = os.path.join('checkpoint', f'{checkpoint_path}_test.pth')
            if not test:
                model.save_checkpoint(path=path)
            print(f"Best test results: reward = {best_reward_test}, best served demand = {test_served_demand}, best rebalancing cost = {test_rebalancing_cost}")