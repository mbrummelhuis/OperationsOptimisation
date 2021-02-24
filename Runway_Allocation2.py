"""
Gurobi algorithm to solve the MILP Runway Allocation problem for the AE4441-16 Operation Optimisation assignment.

It takes takes the generated landing requests as input for the solver


NOTES:
    > Added delay list and added it to the decision variable x
    > Added a penalty for any delay, as of right now this is arbitrary, could/should be tuned.
    > How to correctly implement

CHANGES:
    > Modified datageneration to output IAF directly, took out code for flights_filtered to handle this change in datageneration
    > Added the part where the flights dataframe is enhanced with landing time at specific runway
    > Took out (old) separation constraint, model is feasible without
    > 
"""

import os
import time

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB, LinExpr

# Get path to current folder and select the correct file
cwd = os.getcwd()
filename = 'landing_requests.csv'

startTime = time.time()

# Load the flight data
flights = pd.read_csv(os.path.join(cwd, filename), delimiter=',', skiprows=2, header=0)

# Stored constants------------------------------------------------------------------------------------------
runways, runway_headings, runway_pop = gp.multidict({
    'R18R': [180, 14500],
    'R06E': [60, 700],
    'R24W': [240, 13600]
})

headwind_runways = [h + 180 if h + 180 <= 360 else h - 180 for h in runway_headings.values()]

IAFs, time_to_r, time_to_taxi = gp.multidict({
    'ARTIP': [[1177, 503, 766], [60, 39, 154]],
    'RIVER': [[636, 1177, 1261], [60, 39, 154]],
    'SUGOL': [[709, 1040, 777], [60, 39, 154]]
})
print(time_to_r['ARTIP'][1])

AC_type, AC_fuel, AC_db = gp.multidict({
    'H': [1.029, 17],
    'M': [0.242, 7]
})

# Adding landing time at specific runway to the flights dataframe (for checking the separation constraint)
flights['landing time R18R'] = 0
flights['landing time R06E'] = 0
flights['landing time R24W'] = 0

for f in range(len(flights)):
    for IAF in range(len(IAFs)):  # Iterate over IAFS
        if IAFs[IAF] == flights['IAF'][f]:
            flights['landing time R18R'][f] = time_to_r[IAFs[IAF]][0] + flights['time in seconds'][f]
            flights['landing time R06E'][f] = time_to_r[IAFs[IAF]][1] + flights['time in seconds'][f]
            flights['landing time R24W'][f] = time_to_r[IAFs[IAF]][2] + flights['time in seconds'][f]

print(flights)
'''
Creating a dataframe which contains:
    - IAF
    - Runway
    - Cost for that arc for that AC
    - AC
'''
cpa = []
for AC in AC_type:
    for n in range(len(IAFs)):
        for r in range(len(runways)):
            cpa.append([IAFs[n], runways[r], time_to_r[IAFs[n]][r] * AC_fuel[AC], AC])

# Dataframe with the cost per arc
df_cpa = pd.DataFrame(cpa)
df_cpa.columns = ['IAF', 'Runway', 'CPA/AC', 'AC']

'''
Creating runway compatibility based on wind direction and runway heading
'''
runway_comp = np.zeros((len(flights['wind direction']), len(runways)))

for w in range(len(flights['wind direction'])):
    for h in range(len(headwind_runways)):
        ub = headwind_runways[h] + 90
        lb = headwind_runways[h] - 90
        if lb <= flights['wind direction'][w] < ub:
            runway_comp[w, h] = 1

'''
Setting up the decision variables:
    > flight_{f, r, d}: arriving flight from IAF f to runway r, with delay d
    > G_{n}: noise penalty
'''
x = {}
G = {}

model = gp.Model('runway_allocation')  # Initiate model

dt = 60  # delta time delay in seconds
delay_steps = 10  # maximum number of delay steps
d1 = 0
delays = np.ones(delay_steps).tolist()
delays[0] = d1
for i in range(len(delays) - 1):
    delays[i + 1] = delays[i] + dt

for f in range(len(flights['IAF'])):
    for r in range(len(runways)):
        for d in delays:
            x[f, r, d] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x%s_%s_%s' % (f, r, d))

for r in range(len(runways)):
    G[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='G%s' % (r))

model.update()

'''
Adding the constraints:
    > Always Assign Flight: make sure that every AC gets assigned
    > Wind Restriction: AC should choose to land at runway where the wind is opposite of the heading
    > AC Landing sep: make sure that there is enough time between consecutive arrivals
    > Noise limit switching: when a certain noise limit is surpassed, this constraint files an extra penalty
'''

# Always Assign Flight Constraint
for f in range(len(flights['IAF'])):
    AAS_LHS = LinExpr()
    for r in range(len(runways)):
        for d in delays:
            AAS_LHS += x[f, r, d]

    model.addConstr(lhs=AAS_LHS, sense=GRB.EQUAL, rhs=1, name=f'flight_{f}')

# Wind Restriction Constraint
# for f in range(len(flights)):
#     WR_LHS = LinExpr()
#     for r in range(len(runways)):
#         for d in range(len(delays)):
#
#             if runway_comp[f, r]:
#                 WR_LHS += runway_comp[f, r] * x[f, r, d]
#
#     model.addConstr(lhs=WR_LHS, sense=GRB.EQUAL, rhs=1, name=f'WR_f{f}_r{r}')

# AC Time Separation Constraint
T_sep = 120  # Time separation in seconds

for f1 in range(len(flights['time in seconds'])):
    for f2 in range(len(flights['time in seconds'])):

        if f1 != f2:  # Only calculate when it concerns two different flights

            arrival_f1 = flights['time in seconds'][f1]
            arrival_f2 = flights['time in seconds'][f2]

            for r1 in range(len(runways)):
                for r2 in range(len(runways)):

                    if r1 == r2:  # Only calculate if they might land on the same runway

                        # Only check if the arrival times of both aircraft is within the greatest possible delay.
                        if abs(arrival_f1 - arrival_f2) <= delays[-1]:

                            for delay in delays:
                                AC_sep_LHS = LinExpr()

                                # Time separation is arrival time of f2 - arrival time of f1 + taxi time f1
                                # Second AC is going to be delayed
                                AC_sep_LHS += abs(arrival_f2 + delay - (arrival_f1 +
                                                                        time_to_taxi[flights['IAF'][f1]][r1])) * x[
                                                  f2, r1, delay]

                                model.addConstr(lhs=AC_sep_LHS, sense=GRB.GREATER_EQUAL, rhs=T_sep,
                                                name=f'AC_sep_f{f2}_f{f1}_r{r1}')

# Noise limit switching constraint
# TODO: right now this does nothing with the noise levels generated by the AC type > only in the cost func
# N_limit = np.exp(11/4)

# for f in range(len(flights['approach direction'])):
#     for r in range(len(runways)):
#         NLSC_LHS = LinExpr()
#
#         big_M = 100000
#
#         NLSC_LHS += runway_pop[runways[r]] * x[f, r] - big_M * G[r]

# model.addConstr(lhs=NLSC_LHS, sense=GRB.LESS_EQUAL, rhs=N_limit, name=f'NLSC_f{f}_r{r}_G{r}')

model.update()

'''
Generating the cost function
 > Minimize the cost
 > Cost increases as delay increases
'''

alpha = 0.3
beta = 1 - alpha

n_f = 1
n_n = 1

obj_func = LinExpr()

for t in AC_type:
    for f in range(len(flights['IAF'])):
        for r in range(len(runways)):
            delay_coef = 1
            for d in delays:
                C_F = delay_coef * df_cpa.loc[(df_cpa['IAF'] == flights['IAF'][f]) & (df_cpa['AC'] == t)
                                              & (df_cpa['Runway'] == runways[r]), 'CPA/AC'].values[0]

                obj_func += C_F * x[f, r, d]

                delay_coef += 0.1

# for t in AC_type:
#     for r in range(len(runways)):
#         C_N = AC_db[t]
#
#         obj_func += C_N * G[r]

model.update()
model.setObjective(obj_func, GRB.MINIMIZE)

model.update()
model.write('model_formulation.lp')

model.optimize()

# If the model is feasible, no IIS can be computed
try:
    model.computeIIS()
except:
    pass

# model.remove(gp.Constr())

# model.remove(gp.Constr(AAS_LHS[0]))
model.update()
model.write('model_delete_var.lp')
model.optimize()
try:
    model.computeIIS()
except:
    pass

endTime = time.time()

model.write('model.ilp')
