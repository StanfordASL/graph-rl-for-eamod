"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""
from array import array
from cmath import inf
from collections import defaultdict
from email.charset import add_charset
from itertools import count
import gurobipy as gp
import numpy as np
import subprocess
import torch
import os
import math 
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import json


class AMoD:
    # initialization
    # updated to take scenario
    def __init__(self, scenario):
        if scenario.EV == True:
            # I changed it to deep copy so that the scenario input is not modified by env
            self.scenario = deepcopy(scenario)
            # Road Graph: node - node, edge - connection of node, node attr: 'accInit', edge attr: 'time'
            self.G = scenario.G
            self.G_spatial = scenario.G_spatial
            self.rebTime = self.scenario.rebTime
            self.time = 0  # current time
            self.tf = scenario.tf  # final time
            self.demand = defaultdict(dict)  # demand
            self.depDemand = dict()
            self.arrDemand = dict()
            self.nodes = list(self.G.nodes)
            self.nodes_spatial = list(self.G_spatial.nodes)
            self.gcn_edge_idx = None
            self.gcn_edge_idx_spatial = None
            self.number_nodes = len(self.nodes)  # number of nodes
            self.number_nodes_spatial = len(self.nodes_spatial)
            self.region = range(scenario.spatial_nodes)  # set of regions
            for i in self.region:
                self.depDemand[i] = defaultdict(float)
                self.arrDemand[i] = defaultdict(float)

            self.price = defaultdict(dict)  # price
            self.demand = self.scenario.demand_input
            self.price = self.scenario.p
            # number of vehicles within each node, key: i - node, t - time
            self.acc = defaultdict(dict)
            # number of vehicles arriving at each node, key: i - node, t - time
            self.dacc = defaultdict(dict)
            # number of vehicles within each spatial node, key: i - node, t - time
            self.acc_spatial = defaultdict(dict)
            self.n_charging_vehicles_spatial = defaultdict(dict)
            self.n_rebal_vehicles_spatial = defaultdict(dict)
            self.n_customer_vehicles_spatial = defaultdict(dict)
            
            # number of vehicles arriving at each spatial node, key: i - node, t - time
            self.dacc_spatial = defaultdict(dict)
            # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
            self.rebFlow = defaultdict(dict)
            # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
            self.paxFlow = defaultdict(dict)
            self.edges = list(self.G.edges)
            self.edges_spatial = list(self.G_spatial.edges)
            # map from od regions to road edges connecting them
            self.map_o_d_regions_to_pax_edges = None
            self.charging_edges = None  # edges only used for rebal
            self.map_node_to_outgoing_edges = None  # maps node to outgoing edges
            self.map_node_to_incoming_edges = None  # maps node to incoming edges
            self.create_edge_maps()
            self.create_edge_idx_and_weights()
            self.reset_cars_charging()

            for i, j in self.G.edges:
                self.rebFlow[i, j] = defaultdict(float)
                self.paxFlow[i, j] = defaultdict(float)

            for n in self.nodes:
                self.acc[n][0] = self.G.nodes[n]['accInit']
                self.dacc[n] = defaultdict(float)
            for n in self.nodes_spatial:
                self.acc_spatial[n][0] = self.G_spatial.nodes[n]['accInit']
                self.n_charging_vehicles_spatial[n][0] = 0
                self.n_rebal_vehicles_spatial[n][0] = 0
                self.n_customer_vehicles_spatial[n][0] = 0
                self.dacc_spatial[n] = defaultdict(float)
            self.servedDemand = defaultdict(dict)
            for i, j in self.demand:
                self.servedDemand[i, j] = defaultdict(float)

            self.N = len(self.nodes)  # total number of cells

            # add the initialization of info here
            self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost',
                                      'operating_cost', 'charge_rebalancing_cost', 'spatial_rebalancing_cost'], 0)
            self.reward = 0
            # observation: current vehicle distribution, time, future arrivals, demand
            self.obs = (self.acc, self.time, self.dacc, self.demand)
            self.obs_spatial = (self.acc_spatial, self.time,
                                self.dacc_spatial, self.demand)

    def create_edge_maps(self):
        self.map_o_d_regions_to_pax_edges = dict([])
        self.charging_edges = []
        self.map_region_to_charge_edges = dict([])
        self.map_node_to_outgoing_edges = dict([])
        self.map_node_to_incoming_edges = dict([])

        for node in self.nodes:
            self.map_node_to_incoming_edges[node] = []
            self.map_node_to_outgoing_edges[node] = []

        for o_region in self.region:
            self.map_region_to_charge_edges[o_region] = []
            for d_region in self.region:
                self.map_o_d_regions_to_pax_edges[(o_region, d_region)] = []

        for e in range(len(self.edges)):
            o, d = self.edges[e]
            self.map_node_to_outgoing_edges[o].append(e)
            self.map_node_to_incoming_edges[d].append(e)
            o_region, o_charge = o
            d_region, d_charge = d
            energy_distance = self.scenario.energy_distance[o_region, d_region]
            if (o_charge - d_charge) == energy_distance:
                self.map_o_d_regions_to_pax_edges[(
                    o_region, d_region)].append(e)
            else:
                self.map_region_to_charge_edges[o_region].append(e)
                self.charging_edges.append(e)


    def create_edge_idx_and_weights(self):
        edge_idx = torch.tensor([[], []], dtype=torch.long)
        edge_idx_spatial = torch.tensor([[], []], dtype=torch.long)
        for e_spatial in self.edges_spatial:
            origin_node_idx = self.nodes_spatial.index(e_spatial[0])
            destination_node_idx = self.nodes_spatial.index(e_spatial[1])
            new_edge = torch.tensor(
                [[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx_spatial = torch.cat((edge_idx, new_edge), 1)
        self.gcn_edge_idx_spatial = edge_idx_spatial
        for e in self.edges:
            origin_node_idx = self.nodes.index(e[0])
            destination_node_idx = self.nodes.index(e[1])
            new_edge = torch.tensor(
                [[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx = torch.cat((edge_idx, new_edge), 1)
        self.gcn_edge_idx = edge_idx

    # pax step
    def pax_step(self, paxAction=None, pax_flows_solver=None, episode=1):
       t = self.time
       self.reward = 0
       new_customer_vehicles = 0
       for n in self.nodes:
            self.acc[n][t+1] = self.acc[n][t]
       for n_spatial in self.nodes_spatial:
            self.acc_spatial[n_spatial][t+1] = self.acc_spatial[n_spatial][t]
            assert self.acc_spatial[n_spatial][t] >= -1e-8
            self.n_charging_vehicles_spatial[n_spatial][t+1] = self.n_charging_vehicles_spatial[n_spatial][t]
            assert self.n_charging_vehicles_spatial[n_spatial][t] >= -1e-8
            self.n_rebal_vehicles_spatial[n_spatial][t+1] = self.n_rebal_vehicles_spatial[n_spatial][t]
            assert self.n_rebal_vehicles_spatial[n_spatial][t] >= -1e-8
            self.n_customer_vehicles_spatial[n_spatial][t+1] = self.n_customer_vehicles_spatial[n_spatial][t]
            assert self.n_customer_vehicles_spatial[n_spatial][t] >= -1e-8
       self.info['served_demand'] = 0 # initialize served demand
       self.info["operating_cost"] = 0 # initialize operating cost
       self.info['revenue'] = 0
       self.info['rebalancing_cost'] = 0
       if paxAction is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
           paxAction = pax_flows_solver.optimize()
       self.paxAction = paxAction
    #    if (episode % 1000 == 0):
    #     for i in range(len(self.edges)):
    #         print(str(self.edges[i]) + " paxAction: " + str(paxAction[i]))
       # serving passengers
       satisfied_demand = np.zeros(self.number_nodes_spatial)
       total_demand = np.zeros(self.number_nodes_spatial)
       for origin in range(self.number_nodes_spatial):
            for destination in range(self.number_nodes_spatial):
                total_demand[origin] += self.demand[origin, destination][t]

       for k in range(len(self.edges)):
           i,j = self.edges[k]
           i_region = i[0]
           j_region = j[0]
           if (i_region,j_region) not in self.demand or t not in self.demand[i_region,j_region] or self.paxAction[k] < 1e-3 or i[1]<j[1]:
                continue
           # I moved the min operator above, since we want paxFlow to be consistent with paxAction
           
           # assert paxAction[k] < self.acc[i][t+1] + 1e-3
           self.paxAction[k] = min(self.acc[i][t+1], self.paxAction[k] + 1e-3)
           
           # assert paxAction[k] >= 0
           self.paxAction[k] = max(0, self.paxAction[k])
           
           if satisfied_demand[i_region] + self.paxAction[k] > total_demand[i_region]:
               self.paxAction[k] = total_demand[i_region] - satisfied_demand[i_region]
           satisfied_demand[i_region] += self.paxAction[k]

           self.servedDemand[i_region,j_region][t] += self.paxAction[k] 
           self.paxFlow[i,j][t+self.G.edges[i,j]['time'][self.time]] = self.paxAction[k]
           self.info["operating_cost"] += (self.G.edges[i,j]['time'][self.time]+ self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.paxAction[k]
           self.acc[i][t+1] -= self.paxAction[k]
           self.acc_spatial[i[0]][t+1] -= self.paxAction[k]
           self.n_customer_vehicles_spatial[i[0]][t+1] += self.paxAction[k]
           self.info['served_demand'] += self.paxAction[k]
           self.dacc[j][t+self.G.edges[i,j]['time'][self.time]+self.scenario.time_normalizer] += self.paxFlow[i,j][t+self.G.edges[i,j]['time'][self.time]]
           self.dacc_spatial[j_region][t+self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer] += self.paxFlow[i, j][t+self.G.edges[i, j]['time'][self.time]]
           self.reward += self.paxAction[k]*(self.price[i_region, j_region][t] - (self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep)
           self.info['revenue'] += self.paxAction[k]*(self.price[i_region,j_region][t])
       
       test_spatial_acc_count = np.zeros(self.number_nodes_spatial)
       for n in self.nodes:
            test_spatial_acc_count[n[0]] += self.acc[n][t+1]
       for region in self.nodes_spatial:
            assert abs(test_spatial_acc_count[region] - self.acc_spatial[region][t+1]) < 1e-3
            assert satisfied_demand[region] - total_demand[region] < 1e-3

       self.obs = (self.acc, self.time, self.dacc, self.demand) # for acc, the time index would be t+1, but for demand, the time index would be t
       self.obs_spatial = (self.acc_spatial, self.time, self.dacc_spatial, self.demand)
       done = False # if passenger matching is executed first

       return self.obs, max(0,self.reward), done, self.info
    
    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0 # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        rebreward_internal = 0
        self.rebAction = rebAction      
        # rebalancing
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                assert False
            # update the number of vehicles
            assert rebAction[k] < self.acc[i][t+1] + 1e-3
            if rebAction[k] < 1e-3:
                continue
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k])
            
            self.rebFlow[i,j][t+self.G.edges[i,j]['time'][self.time]] = self.rebAction[k]
            self.dacc[j][t+self.G.edges[i,j]['time'][self.time]+self.scenario.time_normalizer] += self.rebFlow[i,j][t+self.G.edges[i,j]['time'][self.time]]
            self.dacc_spatial[j[0]][t+self.G.edges[i,j]['time'][self.time]+self.scenario.time_normalizer] += self.rebFlow[i,j][t+self.G.edges[i,j]['time'][self.time]]
            self.acc[i][t+1] -= self.rebAction[k] 
            self.acc_spatial[i[0]][t+1] -= self.rebAction[k]
            # charging edge
            if i[1] < j[1] and self.rebAction[k] > 0 and i[0] == j[0]:
                charge_difference = j[1] - i[1]
                # assert charge_difference > 0
                charge_time = math.ceil(charge_difference/self.scenario.charge_levels_per_charge_step)
                # assert charge_time > 0
                avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time])
                # assert avg_energy_price > 0
                self.info['rebalancing_cost'] += avg_energy_price * self.rebAction[k]*charge_difference
                # charge cost negatively influences the reward
                self.reward -= avg_energy_price * self.rebAction[k]*charge_difference
                # we have to add plus one because charging starts in the next timestep
                for future_time in range(t+1, t+charge_time+1):
                    # Calculate potential number of cars charging after adding rebAction[k]
                    potential_cars_charging = self.scenario.cars_charging_per_station[i[0]][future_time] + self.rebAction[k]
                    # Check if this potential number exceeds the station's capacity
                    if potential_cars_charging - self.scenario.cars_per_station_capacity[i[0]] < 1e-7:
                        # If not, proceed with charging
                        self.scenario.cars_charging_per_station[i[0]][future_time] = potential_cars_charging
                        self.n_charging_vehicles_spatial[i[0]][future_time] += self.rebAction[k]
                    else:
                        # If it does, charge only the number of cars that the station can accommodate
                        available_space = self.scenario.cars_per_station_capacity[i[0]] - self.scenario.cars_charging_per_station[i[0]][future_time]
                        self.scenario.cars_charging_per_station[i[0]][future_time] += available_space
                        self.n_charging_vehicles_spatial[i[0]][future_time] += available_space
                    # self.scenario.cars_charging_per_station[i[0]][future_time] += self.rebAction[k]
                    
                    assert self.scenario.cars_charging_per_station[i[0]][future_time] - self.scenario.cars_per_station_capacity[i[0]] < 1e-7
                    self.n_charging_vehicles_spatial[i[0]][future_time] += self.rebAction[k]
            
            # road and charging edge
            elif i[1] - self.scenario.energy_distance[i[0], j[0]] < j[1] and i[0] != j[0] and self.rebAction[k] > 0:
                self.n_rebal_vehicles_spatial[i[0]][t+1] += self.rebAction[k]
                charge_difference = j[1] - i[1] + self.scenario.energy_distance[i[0], j[0]]
                # assert charge_difference > 0
                charge_time = math.ceil(charge_difference/self.scenario.charge_levels_per_charge_step)
                # assert charge_time > 0
                avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time])
                # assert avg_energy_price > 0

                # assert ((self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]) > 0
    
                self.info['spatial_rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info["operating_cost"] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info['rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer - charge_time)*self.scenario.operational_cost_per_timestep*self.rebAction[k] + avg_energy_price * self.rebAction[k]*charge_difference
                
                # we have to add plus one because charging starts in the next timestep
                for future_time in range(t+1, t+charge_time+1):
                    self.scenario.cars_charging_per_station[i[0]][future_time] += self.rebAction[k]
                    assert self.scenario.cars_charging_per_station[i[0]][future_time] - self.scenario.cars_per_station_capacity[i[0]] < 1e-7
                    self.n_charging_vehicles_spatial[i[0]][future_time] += self.rebAction[k]
                self.reward -= avg_energy_price * self.rebAction[k]*charge_difference + (self.G.edges[i,j]['time'][self.time]+self.scenario.time_normalizer - charge_time)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
            # road edge
            elif self.rebAction[k] > 0:
                self.n_rebal_vehicles_spatial[i[0]][t+1] += self.rebAction[k]

                # assert ((self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]) > 0

                self.info['rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info['spatial_rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info["operating_cost"] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                
                self.reward -= (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]

        
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing  
        for k in range(len(self.edges)):
            o, d = self.edges[k]
            if (o, d) in self.rebFlow and t in self.rebFlow[o, d]:
                self.acc[d][t+1] += self.rebFlow[o, d][t]
                self.acc_spatial[d[0]][t+1] += self.rebFlow[o, d][t]
                # check if charging capacity has freed up
                if d[1] > o[1] and o[0] == d[0]:
                    continue
                else:
                    self.n_rebal_vehicles_spatial[o[0]][t+1] -= self.rebFlow[o, d][t]

            if (o, d) in self.paxFlow and t in self.paxFlow[o, d]:
                self.acc[d][t+1] += self.paxFlow[o, d][t]
                self.acc_spatial[d[0]][t+1] += self.paxFlow[o, d][t]
                self.n_customer_vehicles_spatial[o[0]][t+1] -= self.paxFlow[o, d][t]
        
        # rebreward_internal takes into account a penalty for vehicles
        # that are not above 30% charge in subsequent step

        # -5 penalty for vehicles in bottom third of charge
        # otherwise, +5 * fraction of charge per vehicle 

        charging_penalty = 0
        charge_limit = math.ceil(self.scenario.number_charge_levels*0.3)
        for c in range(charge_limit):
            for region in self.nodes_spatial:
                charging_penalty += self.acc[(region, c)][self.time+1] * (-5)
        
        for c in range(charge_limit, self.scenario.number_charge_levels):
            for region in self.nodes_spatial:
                charging_penalty += self.acc[(region, c)][self.time+1] * (5) * ((c)/(self.scenario.number_charge_levels))

        wasted_customers_penalty = 0
        for region in self.nodes_spatial:
            demand = 0
            for region2 in self.nodes_spatial:
                demand += self.demand[region, region2][self.time+1]
            unserved_demand = demand - self.acc_spatial[region][self.time+1]
            wasted_customers_penalty += min(0, unserved_demand * (-10))
        
        overconcentration_penalty = 0
        total_vehicles = sum(self.acc[(region, charge_level)][self.time+1] for region in self.nodes_spatial for charge_level in range(self.scenario.number_charge_levels))

        for region in self.nodes_spatial:
            total_vehicles_region = sum(self.acc[(region, charge_level)][self.time+1] for charge_level in range(self.scenario.number_charge_levels))
            if total_vehicles_region >= 0.8 * total_vehicles:
                overconcentration_penalty -= 1000
        

        # print("reward: " + str(self.reward))
        # print("charging_penalty: " + str(charging_penalty))
        # print("wasted_customers_penalty: " + str(wasted_customers_penalty))
        
        rebreward_internal = self.reward + charging_penalty + wasted_customers_penalty + overconcentration_penalty
            
        self.time += 1
        # use self.time to index the next time step
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        done = (self.tf == t+1) # if the episode is completed

        return self.obs, self.reward, rebreward_internal, done, self.info
    
    def reset(self, bool_sample_demand=True, seed=None):
        # reset the episode
        self.acc = defaultdict(dict)
        self.acc_spatial = defaultdict(dict)
        self.n_rebal_vehicles_spatial = defaultdict(dict)
        self.n_customer_vehicles_spatial = defaultdict(dict)
        
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        self.reset_cars_charging()

        tripAttr = self.scenario.get_random_demand(bool_sample_demand, seed=seed)
        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        for n_spatial in self.nodes_spatial:
            self.acc_spatial[n_spatial][0] = self.G_spatial.nodes[n_spatial]['accInit']
            self.dacc_spatial[n_spatial] = defaultdict(float)
            self.n_rebal_vehicles_spatial[n_spatial][0] = 0
            self.n_customer_vehicles_spatial[n_spatial][0] = 0
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs

    def reset_cars_charging(self):
        self.scenario.cars_charging_per_station = defaultdict(dict)
        self.n_charging_vehicles_spatial = defaultdict(dict)
        for region in range(self.scenario.spatial_nodes):
            self.scenario.cars_charging_per_station[region] = defaultdict(float)
            self.n_charging_vehicles_spatial[region] = defaultdict(float)
            for t in range(self.scenario.tf):
                self.scenario.cars_charging_per_station[region][t] = 0.
                self.n_charging_vehicles_spatial[region][t] = 0.



class Scenario:
    def __init__(self, EV=True, spatial_nodes=4, charging_stations=None, cars_per_station_capacity=None, number_charge_levels=10, charge_levels_per_charge_step=1, energy_distance=None, tf=60, sd=None, tripAttr=None,
                 demand_ratio=None, trip_length_preference=0.25, grid_travel_time=1, reb_time=None, total_acc=None, p_energy=None, time_granularity=0.5, operational_cost_per_timestep=0.5):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each nodes, 
        #          float/int - total demand out of each nodes satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        self.EV = EV
        if sd != None:
            np.random.seed(self.sd)
        
        if EV == True: 
            self.time_normalizer = 1 
            self.time_granularity = time_granularity
            self.operational_cost_per_timestep = operational_cost_per_timestep
            self.spatial_nodes = spatial_nodes
            self.charging_stations = charging_stations
            self.cars_per_station_capacity = cars_per_station_capacity
            self.cars_charging_per_station = defaultdict(dict)
            self.number_charge_levels = number_charge_levels
            self.charge_levels_per_charge_step = charge_levels_per_charge_step
            self.energy_distance = energy_distance
            self.p_energy = np.array(p_energy)  # price of energy in $/kWh
            self.intermediate_charging_station = defaultdict(dict)
            self.time = 0  # current time
            self.is_json = False
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.tf = tf
            self.G = nx.empty_graph()
            self.G = self.G.to_directed()
            self.G_spatial = nx.empty_graph()
            self.G_spatial = self.G_spatial.to_directed()

            self.demand_input, self.p, self.rebTime = defaultdict(dict), defaultdict(dict), defaultdict(dict)
            for item in tripAttr:
                t,o,d,v,p = item['time_stamp'], item['origin'], item['destination'], item['demand'], item['price']
                if (o,d) not in self.demand_input:
                    self.demand_input[o,d], self.p[o,d] = defaultdict(float), defaultdict(float)
                self.demand_input[o,d][t] += v*demand_ratio
                self.p[o,d][t] += p*demand_ratio
            
            for item in reb_time:
                hr,o,d,rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
                for t in range(0,tf+1):
                    # self.rebTime[o,d][t] = max(int(round(rt)),1) used to be this
                    self.rebTime[o,d][t] = int(round(rt))
            
            #print("EDGES IN AMOND_ENV:")
            # add charge edges
            self.add_charge_edges()

            # add road edges
            self.add_road_edges()
            
            # add artificial edges (commented out due to decoupling of GNN and AMoD_Env)
            # for o_node in list(self.G.nodes):
            #     for d_node in list(self.G.nodes):
            #         o_region = o_node[0]
            #         o_charge = o_node[1]
            #         d_region = d_node[0]
            #         d_charge = d_node[1]
            #         energy_dist = self.energy_distance[o_region,d_region]
            #         if o_region == d_region or o_charge - energy_dist >= d_charge: # We already created charge edges and regular road edges
            #             continue
            #         # edges from a charging station
            #         elif self.charging_stations[o_region]:
            #             if (d_charge <= self.number_charge_levels-1 - energy_dist):
            #                 self.add_artificial_edges_from_or_to_station(o_node, d_node)

            self.edges = list(self.G.edges)
            #print("Number of edges: " + str(len(self.edges)))
            #print("Number of spatial nodes: " + str(len(self.G_spatial.nodes)))
            #print("Number of nodes: " + str(len(self.G.nodes)))
            self.tf = tf

            for o,d in self.edges:
                for t in range(0,tf*2):
                    if t in self.demand_input[o[0],d[0]] and self.demand_input[o[0],d[0]][t] > 0:
                        continue
                    else:
                        self.demand_input[o[0],d[0]][t] = 0
                        self.p[o[0],d[0]][t] = 0

            for item in total_acc:
                hr, acc = item['hour'], item['acc']
                for region in self.G_spatial.nodes:
                    self.G_spatial.nodes[region]['accInit'] = int(0)
                    # for c in range(self.number_charge_levels):
                    #     cut_off_charge = int(0.5*self.number_charge_levels)
                    #     print("cutoff charge init", cut_off_charge)
                    #     number_of_used_charges = (self.number_charge_levels-cut_off_charge)
                    #     number_cars_per_node = int(acc/(len(list(self.G_spatial.nodes))*number_of_used_charges))
                    #     if c >= cut_off_charge:
                    #         self.G.nodes[(region,c)]['accInit'] = number_cars_per_node
                    #         self.G_spatial.nodes[region]['accInit'] += number_cars_per_node
                    #     else:
                    #         self.G.nodes[(region,c)]['accInit'] = 0
                    # only bottom 60%
                    for c in range(self.number_charge_levels):
                        cut_off_charge = int(1.*self.number_charge_levels)
                        #print("cutoff charge init", cut_off_charge)
                        number_of_used_charges = cut_off_charge
                        number_cars_per_node = int(acc/(len(list(self.G_spatial.nodes))*number_of_used_charges))
                        if c <= cut_off_charge:
                            self.G.nodes[(region,c)]['accInit'] = number_cars_per_node
                            self.G_spatial.nodes[region]['accInit'] += number_cars_per_node
                        else:
                            self.G.nodes[(region,c)]['accInit'] = 0
                    # for c in range(self.number_charge_levels):
                    #     number_cars_per_node = int(acc/(len(list(self.G_spatial.nodes))))
                    #     if c == number_charge_levels - 1:
                    #         self.G.nodes[(region,c)]['accInit'] = number_cars_per_node
                    #         self.G_spatial.nodes[region]['accInit'] += number_cars_per_node
                    #     else:
                    #         self.G.nodes[(region,c)]['accInit'] = 0

                        
                break  # only need the first time step, if I want variable acc, I need to change this
            self.tripAttr = self.get_random_demand() # randomly generated demand

    def add_charge_edges(self):
        counter = 0
        for l in range(self.spatial_nodes):
            if not self.charging_stations[l]:
                continue
            for c1 in range(self.number_charge_levels - 1):
                # old version in codebase
                # fully_charged = c1 == (self.number_charge_levels-1)
                # c2 = c1
                # while not fully_charged:
                #     c2 += self.charge_levels_per_charge_step
                #     if c2 >= self.number_charge_levels:
                #         c2 = (self.number_charge_levels-1)
                #         fully_charged = True
                #     assert c1 >= 0 and c2 > c1 and c2 < self.number_charge_levels
                #     self.G.add_edge((l, c1), (l, c2))
                #     print("edge: " + str(counter) + " --->  l: " + str(l) + " c1: " + str(c1) + " c2: " + str(c2))
                #     counter += 1
                #     self.G.edges[(l, c1), (l, c2)]['time'] = dict()
                #     for t in range(0, self.tf+1):
                #         self.G.edges[(l, c1), (l, c2)]['time'][t] = math.ceil((c2-c1)/self.charge_levels_per_charge_step) - self.time_normalizer
                
                # version assumed to be right
                c2 = c1 + self.charge_levels_per_charge_step 
                if c2 >= self.number_charge_levels:
                    c2 = (self.number_charge_levels-1)
                assert c1 >= 0 and c2 > c1 and c2 < self.number_charge_levels
                self.G.add_edge((l, c1), (l, c2))
                #print("edge: " + str(counter) + " --->  l: " + str(l) + " c1: " + str(c1) + " c2: " + str(c2))
                counter += 1
                self.G.edges[(l, c1), (l, c2)]['time'] = dict()
                for t in range(0, self.tf+1):
                    self.G.edges[(l, c1), (l, c2)]['time'][t] = math.ceil((c2-c1)/self.charge_levels_per_charge_step) - self.time_normalizer
    
    def add_road_edges(self):
        counter = 0
        for o in range(self.spatial_nodes):
            for d in range(self.spatial_nodes):
                self.G_spatial.add_edge(o, d)
                self.G_spatial.edges[o, d]['time'] = dict()
                for t in range(0, self.tf+1):
                    self.G_spatial.edges[o, d]['time'][t] = math.ceil(self.rebTime[o, d][t]) - self.time_normalizer
                for c in reversed(range(self.number_charge_levels)):
                    # removes top and bottom node for nodes without charge stations -> removes infeasible edges
                    target_charge = int(c - self.energy_distance[o, d])
                    if (not self.charging_stations[o]) and (c == self.number_charge_levels-1):
                        continue
                    elif (target_charge < 0):
                        break
                    elif (not self.charging_stations[d]) and (target_charge == 0):
                        break

                    # old version from codebase
                    # assert target_charge < c  # we have to lose energy to move

                    # version assumed to be correct
                    if (o == d):
                        target_charge = c # no charge lost if staying at the same location
                    else:
                        assert target_charge < c  # we have to lose energy to move
                    
                    if (o != d):
                        self.G.add_edge((o, c), (d, target_charge))
                        #print("edge: " + str(counter) + " --->  o: " + str(o) + " d: " + str(d) + " c1: " + str(c) + ' c2: ' + str(target_charge))
                        counter += 1
                        self.G.edges[(o, c), (d, target_charge)]['time'] = dict()
                        for t in range(0, self.tf+1):
                            self.G.edges[(o, c), (d, target_charge)]['time'][t] = math.ceil(self.rebTime[o, d][t]) - self.time_normalizer

    def add_artificial_edges_from_or_to_station(self, o_node: tuple, d_node: tuple):
        counter = 0
        o_region = o_node[0]
        d_region = d_node[0]
        target_energy = d_node[1] + self.energy_distance[o_region,d_region]
        energy_dist = (target_energy - o_node[1])
        if energy_dist % self.charge_levels_per_charge_step != 0:
            return
        energy_time = math.ceil(energy_dist/self.charge_levels_per_charge_step)
        self.G.add_edge(o_node, d_node)
        #print("edge " + str(counter) + " --> o: " + str(o_node[0]) + " d: " + str(d_node[0]) + " c1: " + str(o_node[1]) + " c2: " + str(d_node[1]))
        counter += 1
        self.G.edges[o_node, d_node]['time'] = dict()
        for t in range(0,self.tf+1):
            if t+energy_time < self.tf:
                spatial_time = self.rebTime[o_region, d_region][t+energy_time]
            else:
                spatial_time = self.rebTime[o_region, d_region][t]
            total_distance = spatial_time + energy_time
            self.G.edges[o_node,d_node]['time'][t] = math.ceil(total_distance) - self.time_normalizer
        

    # gaussian prediction = mean is ground truth and variance is metric of randomness of demand (5, 10, etc)
    
    def get_random_demand(self, bool_random = True, seed=None):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        if seed != None and bool_random:
            np.random.seed(seed)

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.EV:
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    if (i[0], j[0]) in self.demand_input and t in self.demand_input[i[0], j[0]]:
                        if bool_random:
                            demand[i[0],j[0]][t] = np.random.poisson(self.demand_input[i[0],j[0]][t])
                        else:
                            demand[i[0], j[0]][t] = self.demand_input[i[0], j[0]][t]
                        price[i[0], j[0]][t] = self.p[i[0],j[0]][t]  
                    else:
                        demand[i[0], j[0]][t] = 0
                        price[i[0], j[0]][t] = 0
                    tripAttr.append(
                        (i[0], j[0], t, demand[i[0], j[0]][t], price[i[0], j[0]][t]))

        return tripAttr
