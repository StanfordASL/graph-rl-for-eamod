# Class def for optimization
import gurobipy as gp
from gurobipy import quicksum
import numpy as np
import os
import time


class RebalFlowSolver:  
    def __init__(self, env, desiredAcc, gurobi_env):
        # Initialize model
        self.cons_charge_graph1 = {}
        self.cons_charge_graph2 = {}
        self.cons_spatial_graph_charging_cars = {}
        t = env.time
        self.m = gp.Model(env=gurobi_env)

        self.m.Params.Method = 2
        self.m.Params.Crossover = 0
        #self.m.Params.BarConvTol = 1e-12
        self.m.Params.Threads = 32
        self.m.Params.LogToConsole = 0
        #self.m.setParam("LogFile", os.path.join(os.getcwd(), 'reb_flow_gurobi_log.log'))

        self.flow = self.m.addMVar(shape=(len(env.edges)), lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="flow") # both could be INTEGER
        self.slack_variables = self.m.addMVar(shape=(len(env.nodes)), lb=-10000000, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="slack")
        self.slack_variables_abs = self.m.addMVar(shape=(len(env.nodes)), lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="slack_abs")

        for n_idx in range(len(env.nodes)):
            n = env.nodes[n_idx]
            outgoing_edges = env.map_node_to_outgoing_edges[n]
            incoming_edges = env.map_node_to_incoming_edges[n]

            # Constraint 1: We can not have more vehicles flowing out of a node, than vehicles at the node
            self.cons_charge_graph1[n_idx] = self.m.addConstr(sum(self.flow[outgoing_edges]) <= env.acc[n][t + 1])

            # Constraint 2: We want to reach the target distribrution
            self.cons_charge_graph2[n_idx] = self.m.addConstr(sum(self.flow[incoming_edges]) - sum(self.flow[outgoing_edges]) + self.slack_variables[n_idx] == desiredAcc[n] - env.acc[n][t + 1]) 
            self.m.addGenConstrAbs(self.slack_variables_abs[n_idx], self.slack_variables[n_idx], "absconstr")
            
            # Constraint 3: We cannot charge more vehicles then we have charging spots
        for r_idx in range(env.number_nodes_spatial):
            outgoing_charge_edges = env.map_region_to_charge_edges[r_idx]
            self.cons_spatial_graph_charging_cars[r_idx] = self.m.addConstr(sum(self.flow[outgoing_charge_edges]) <= env.scenario.cars_per_station_capacity[r_idx] - env.scenario.cars_charging_per_station[r_idx][t+1])
            # self.cons_spatial_graph_charging_cars[r_idx] = self.m.addConstr(env.n_charging_vehicles_spatial[r_idx][t+1] <= env.scenario.cars_per_station_capacity[r_idx])
        
        self.obj1 = 0
        for n_idx in range(len(env.nodes)):
            self.obj1 += self.slack_variables_abs[n_idx] * 1e10
        # self.obj1 = gp.abs_(quicksum(self.slack_variables) * 1e10)

        self.obj2 = 0
        for e_idx in range(len(env.edges)):
            i,j = env.edges[e_idx]
            self.obj2 += self.flow[e_idx] * (env.G.edges[i,j]['time'][t + 1]+env.scenario.time_normalizer) * env.scenario.operational_cost_per_timestep
        self.m.setObjective(self.obj1+self.obj2, gp.GRB.MINIMIZE)

    def update_constraints(self, desired_acc, env):
        desired_acc_checksum = 0
        acc_checksum = 0
        for n_idx in range(len(env.nodes)):
            node_charge = env.nodes[n_idx]

            desired_acc_checksum += desired_acc[node_charge]
            acc_checksum += env.acc[node_charge][env.time + 1]
            self.cons_charge_graph1[n_idx].RHS = env.acc[node_charge][env.time + 1]
            self.cons_charge_graph2[n_idx].RHS = desired_acc[node_charge] - env.acc[node_charge][env.time + 1]
        assert abs(desired_acc_checksum - acc_checksum) < 1e-5
        for r_idx in range(env.number_nodes_spatial):
            self.cons_spatial_graph_charging_cars[r_idx].RHS = env.scenario.cars_per_station_capacity[r_idx] - env.scenario.cars_charging_per_station[r_idx][env.time+1]
        self.m.update()
        
    def update_objective(self, env):
        time_a = time.time()
        self.obj2 = sum((self.flow[e_idx] * (env.G.edges[env.edges[e_idx][0], env.edges[e_idx][1]]['time'][env.time + 1] + env.scenario.time_normalizer) * env.scenario.operational_cost_per_timestep) for e_idx in range(len(env.edges)))
        time_a_end = time.time() - time_a

        time_b = time.time()
        self.m.setObjective(self.obj1+self.obj2, gp.GRB.MINIMIZE)
        time_b_end = time.time() - time_b

        time_c = time.time()
        self.m.update()
        time_c_end = time.time() - time_c

    # def optimize(self):
    #     self.m.optimize()
    #     if self.m.status == 3:
    #         print("Optimization is infeasible.")
    #     assert self.m.status == 2
    #     action = self.flow.X
    #     return action
    
    def optimize(self):
        self.m.optimize()
        if self.m.status == 3:
            print("Optimization is infeasible.")
            # Return a default flow
            return np.zeros(self.flow.shape)
        elif self.m.status != 2:
            print("Optimization did not complete successfully.")
            return np.zeros(self.flow.shape)  # or handle other statuses as needed
        action = self.flow.X
        return action
        