# Class def for optimization
import gurobipy as gp
import numpy as np
from gurobipy import quicksum

class PaxFlowsSolver:

    def __init__(self, env, gurobi_env):
        # Initialize model
        self.env = env
        self.cons_charge_graph = {}
        self.cons_spatial_graph = {}
        self.cons_rebal_edges = {}
        t = self.env.time
        self.m = gp.Model(env=gurobi_env)
        self.m.Params.LogToConsole = 0
        self.flow = self.m.addMVar(shape=(len(
            self.env.edges)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="flow")

        # constr. 1: can't have outflow more than initial accumulation
        for n in self.env.nodes:
            self.cons_charge_graph[n] = self.m.addConstr(
                sum(self.flow[self.env.map_node_to_outgoing_edges[n]]) <= float(
                    self.env.acc[n][t])
            )

        # constr. 2: no more flow than demand
        for i in self.env.region:
            for j in self.env.region:
                self.cons_spatial_graph[(i, j)] = self.m.addConstr(
                    sum(self.flow[self.env.map_o_d_regions_to_pax_edges[(
                        i, j)]]) <= self.env.demand[i, j][t]
                )
        # constr. 3: pax flow is zero on rebal edges
        self.cons_rebal_edges[0] = self.m.addConstr(
            sum(self.flow[self.env.charging_edges]) == 0
        )
        # objective function
        obj = 0
        for i in range(len(self.env.edges)):
            edge = self.env.edges[i]
            o_region = edge[0][0]
            d_region = edge[1][0]
            obj += self.flow[i] * (self.env.price[o_region, d_region][t] - (self.env.G.edges[edge]['time']
                                   [self.env.time]+self.env.scenario.time_normalizer) * self.env.scenario.operational_cost_per_timestep)
        self.m.setObjective(obj, gp.GRB.MAXIMIZE)

    def update_constraints(self):
        for n in self.env.nodes:
            self.cons_charge_graph[n].RHS = float(
                self.env.acc[n][self.env.time])
        for i in self.env.region:
            for j in self.env.region:
                self.cons_spatial_graph[(
                    i, j)].RHS = self.env.demand[i, j][self.env.time]
        self.m.update()

    def update_objective(self):
        obj = quicksum(self.flow[i] * (self.env.price[self.env.edges[i][0][0], self.env.edges[i][1][0]][self.env.time] - (self.env.G.edges[self.env.edges[i]]
                  ['time'][self.env.time]+self.env.scenario.time_normalizer) * self.env.scenario.operational_cost_per_timestep) for i in range(len(self.env.edges)))
        self.m.setObjective(obj, gp.GRB.MAXIMIZE)
        self.m.update()

    # def optimize(self):
    #     self.m.optimize()
    #     paxAction = self.flow.X
    #     return paxAction
    
    def optimize(self):
        self.m.optimize()
        if self.m.status == 3:
            print("Optimization is infeasible.")
            # Return a default flow
            return np.zeros(self.flow.shape)
        elif self.m.status != 2:
            print("Optimization did not complete successfully.")
            return np.zeros(self.flow.shape)  # or handle other statuses as needed
        paxAction = self.flow.X
        return paxAction
