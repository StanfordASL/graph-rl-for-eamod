import wandb
from gym import spaces
import gym
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
from src.algos.reb_flows_solver import RebalFlowSolver
from src.algos.pax_flows_solver import PaxFlowsSolver
import numpy as np
import sys
import time
sys.path.insert(0, '../../')


class FleetEnv(gym.Env):
    """
    Custom Environment that follows gym interface. 
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, env=None, gurobi_env=None, T=10, scale_factor_reward=0.01, scale_factor=0.0001, price_scale_factor=0.1, test=False):
        super(FleetEnv, self).__init__()
        self.env = env
        self.gurobi_env = gurobi_env
        self.episode = 0
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_rebalancing_cost = 0
        self.T = T
        self.pax_flows_solver = PaxFlowsSolver(
            env=self.env, gurobi_env=gurobi_env)
        desired_acc = {env.nodes[i]: int(self.env.G.nodes[env.nodes[i]]['accInit'])
                       for i in range(env.number_nodes)}  # over nodes
        _, paxreward, done, info_pax = env.pax_step(
            pax_flows_solver=self.pax_flows_solver)
        self.episode_served_demand += info_pax['served_demand']
        self.episode_reward += paxreward
        # definition rebal solver
        self.rebal_flow_solver = RebalFlowSolver(
            env=self.env, desiredAcc=desired_acc, gurobi_env=gurobi_env)
        # Define action and observation space
        self.scale_factor = scale_factor
        self.price_scale_factor = price_scale_factor
        self.scale_factor_reward = scale_factor_reward
        self.initial_state = self.parse_state().astype(np.float32)

        self.action_space = spaces.Box(
            low=0, high=1000, shape=(len(self.env.nodes)*2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=self.initial_state.shape, dtype=np.float32)
        self.test = test
        self.warm_up_phase = 0
        if test:
            self.warm_up_phase = 100
            self.number_episodes_for_test_eval = 100
            self.rewards = np.empty(self.number_episodes_for_test_eval)
            self.times_arr = np.empty(self.number_episodes_for_test_eval)
            wandb.init(
                # Set the project where this run will be logged
                project='e-amod',
                # pass a run name
                name="test_evaluation_stable_baselines",
                # Track hyperparameters and run metadata
                config={
                    "number_chargelevels": env.scenario.number_charge_levels,
                    "number_spatial_nodes": env.scenario.spatial_nodes,
                    "number_vehicles_per_node_init": env.G.nodes[(0, int(self.env.scenario.number_charge_levels*0.3))]['accInit'],
                    "charging_stations": list(env.scenario.charging_stations),
                    "charging_station_capacities": list(env.scenario.cars_per_station_capacity),
                    "scale_factor": scale_factor,
                    "scale_price": price_scale_factor,
                    "time_horizon": self.T,
                    "episode_length": env.tf,
                    "charge_levels_per_timestep": env.scenario.charge_levels_per_charge_step,
                })

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        wandb.log({"Episode": self.episode, "Reward": self.episode_reward,
                  "ServedDemand": self.episode_served_demand, "Reb. Cost": self.episode_rebalancing_cost})
        if self.episode == self.warm_up_phase-1:
                self.times_previous = time.time()
        if self.test and self.episode >= self.warm_up_phase:
            if self.episode - self.warm_up_phase < self.number_episodes_for_test_eval:
                self.rewards[self.episode -
                             self.warm_up_phase] = self.episode_reward
                self.times_arr[self.episode -
                             self.warm_up_phase] = time.time() - self.times_previous
                self.times_previous = time.time()
                if self.episode+1 - self.warm_up_phase == self.number_episodes_for_test_eval:
                    wandb.log({"Mean reward": self.rewards.mean(),
                              "Std Reward": self.rewards.std(),
                              "Mean time": self.times_arr.mean()})
        self.episode += 1
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_rebalancing_cost = 0
        self.env.reset(not self.test)
        self.pax_flows_solver.update_constraints()
        self.pax_flows_solver.update_objective()
        _, paxreward, done, info_pax = self.env.pax_step(
            pax_flows_solver=self.pax_flows_solver)
        self.episode_served_demand += info_pax['served_demand']
        self.episode_reward += paxreward
        return self.initial_state

    def step(self, action):
        a_out_concentration = action[:len(self.env.nodes)]
        a_out_is_zero = action[len(self.env.nodes):]
        jitter = 1e-20
        concentration = F.softplus(torch.tensor(
            a_out_concentration)).reshape(-1) + jitter
        non_zero = torch.sigmoid(torch.tensor(a_out_is_zero)).reshape(-1)

        concentration_without_zeros = torch.tensor([], dtype=torch.float32)
        sampled_zero_bool_arr = []
        log_prob_for_zeros = 0
        for node in range(non_zero.shape[0]):
            sample = torch.bernoulli(non_zero[node])
            if sample > 0:
                indices = torch.tensor([node])
                new_element = torch.index_select(concentration, 0, indices)
                concentration_without_zeros = torch.cat(
                    (concentration_without_zeros, new_element), 0)
                sampled_zero_bool_arr.append(False)
                log_prob_for_zeros += torch.log(non_zero[node])
            else:
                sampled_zero_bool_arr.append(True)
                log_prob_for_zeros += torch.log(1-non_zero[node])
        if concentration_without_zeros.shape[0] != 0:
            m = Dirichlet(concentration_without_zeros)
            dirichlet_action = m.rsample()
            dirichlet_action_np = list(dirichlet_action.detach().numpy())
        action_np = []
        dirichlet_idx = 0
        for node in range(non_zero.shape[0]):
            if sampled_zero_bool_arr[node]:
                action_np.append(0.)
            else:
                action_np.append(dirichlet_action_np[dirichlet_idx])
                dirichlet_idx += 1

        total_idle_acc = sum(
            self.env.acc[n][self.env.time+1] for n in self.env.nodes)
        desired_acc = {self.env.nodes[i]: int(
            action_np[i] * total_idle_acc) for i in range(self.env.number_nodes)}  # over nodes
        total_desiredAcc = sum(desired_acc[n] for n in self.env.nodes)
        missing_cars = total_idle_acc - total_desiredAcc
        most_likely_node = np.argmax(action_np)
        if missing_cars != 0:
            desired_acc[self.env.nodes[most_likely_node]] += missing_cars
            total_desiredAcc = sum(desired_acc[n] for n in self.env.nodes)
        assert abs(total_desiredAcc - total_idle_acc) < 1e-5
        for n in self.env.nodes:
            assert desired_acc[n] >= 0
        # solve minimum rebalancing distance problem (Step 3 in paper)
        self.rebal_flow_solver.update_constraints(desired_acc, self.env)
        self.rebal_flow_solver.update_objective(self.env)

        rebAction = self.rebal_flow_solver.optimize()
        _, reb_reward, reb_reward_internal, done, info = self.env.reb_step(rebAction)
        self.episode_rebalancing_cost += info['rebalancing_cost']
        self.episode_reward += reb_reward
        reward = reb_reward_internal
        if not done:
            self.pax_flows_solver.update_constraints()
            self.pax_flows_solver.update_objective()
            _, paxreward, done, info_pax = self.env.pax_step(
                pax_flows_solver=self.pax_flows_solver)
            self.episode_served_demand += info_pax['served_demand']
            self.episode_reward += paxreward
            reward += paxreward
            state = self.parse_state()
        else:
            state = self.initial_state

        return state.astype(np.float32), self.scale_factor_reward*reward, done, info

    def parse_state(self):
        # is a result of the code below. The state space has 3 dimensions. One is a scalar and 2 are vectros of length T
        input_size = 2*self.T + 2
        x = np.reshape(
            np.squeeze(
                np.concatenate((
                    np.reshape([float(
                        n[1])/self.env.scenario.number_charge_levels for n in self.env.nodes], (1, 1, self.env.number_nodes)),
                    np.reshape([self.env.acc[n][self.env.time+1] *
                               self.scale_factor for n in self.env.nodes], (1, 1, self.env.number_nodes)),
                    np.reshape([[(self.env.acc[n][self.env.time+1] + self.env.dacc[n][t])*self.scale_factor for n in self.env.nodes]
                                for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes)),
                    np.reshape([[sum([self.env.price[o[0], j][t]*self.scale_factor*self.price_scale_factor*(self.env.demand[o[0], j][t])*((o[1]-self.env.scenario.energy_distance[o[0], j]) >= int(not self.env.scenario.charging_stations[j]))
                                      for j in self.env.region]) for o in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes))),
                    axis=1), axis=0), (input_size, self.env.number_nodes)
        )

        return np.transpose(x)

    def close(self):
        pass
