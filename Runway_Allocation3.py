"""
Gurobi algorithm to solve the MILP Runway Allocation problem for the AE4441-16 Operation Optimisation assignment.

It takes takes the generated landing requests as input for the solver

CHANGES:
    > Modified datageneration to output IAF directly, took out code for flights_filtered to handle this change in datageneration
    > Added the part where the flights dataframe is enhanced with landing time at specific runway
    > Took out (old) separation constraint, model is feasible without
    > Made Runway_Allocation3 because I wanted to clean up the code but didn't want to throw away all code that wasn't immediately useful
        > Deleted runway compatibility for wind direction
        > Took out noise penalty from the decision variables
        > Deleted wind restriction and noise limit swithing constraint
    > Reprogrammed separation constraint using the additional data in flights df.

TODO:
    > Validate whether the aircraft separation constraint works properly
    > Do some fancy kind of data visualisation
    > Tune coefficients of the objective function

NOTES Kars:
    > Got some kind of output, but still not really what we'd like.
    > Looking more into it tomorrow morning 18/02/21
"""

import gurobipy as gp
import pandas as pd
from gurobipy import GRB, LinExpr, Model
import numpy as np
import numpy.random as rnd
import time
import os
import csv

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

AC_type, AC_fuel, AC_db = gp.multidict({
    'H': [1.029, 17],
    'M': [0.242, 7]
})

# Adding landing time at specific runway to the flights dataframe (for checking the separation constraint)
flights['landing time R18R'] = 0
flights['landing time R06E'] = 0
flights['landing time R24W'] = 0

for f in range(len(flights)):
    for IAF in range(len(IAFs)): #Iterate over IAFS
        if IAFs[IAF] == flights['IAF'][f]: 
            flights.loc[f,'landing time R18R'] = time_to_r[IAFs[IAF]][0] + flights['time in seconds'][f]
            flights.loc[f,'landing time R06E'] = time_to_r[IAFs[IAF]][1] + flights['time in seconds'][f]
            flights.loc[f,'landing time R24W'] = time_to_r[IAFs[IAF]][2] + flights['time in seconds'][f]

# Print flights for checking      
print(flights)

# Create a dataframe which includes the cost per arc (CPA) from IAF to runway
cpa = []
for AC in AC_type:
    for n in range(len(IAFs)):
        for r in range(len(runways)):
            cpa.append([IAFs[n], runways[r], time_to_r[IAFs[n]][r] * AC_fuel[AC], AC])

df_cpa = pd.DataFrame(cpa)
df_cpa.columns = ['IAF', 'Runway', 'CPA/AC', 'AC']


'''
Setting up the decision variables:
    > flight_{f, r, d}: arriving flight from IAF f to runway r, with delay d
'''
x = {}

model = gp.Model('runway_allocation') #Initiate model

dt = 60 # delta time delay in seconds
delay_steps = 10 #maximum number of delay steps (minutes)
delays = np.ones(delay_steps).tolist()
delays[0] = 0
for i in range(len(delays) - 1):
    delays[i + 1] = delays[i] + dt

for f in range(len(flights['IAF'])):
    for r in range(len(runways)):
        for d in delays:
            x[f, r, d] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x%s_%s_%s' % (f, r, d)) # name='x%s_%s_%s' % (f, r, d)

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

# AC Time Separation Constraint
# For each flight, check if they are landing on the same runway within the T_sep interval. If this is the case, append constraint 
# stating only one flight can land.
# TODO: Figure out how delay fits into all of this
T_sep = 120  # Time separation in seconds

for f1 in range(len(flights['time in seconds'])):
    for f2 in range(len(flights['time in seconds'])):
        if f1 != f2:
            if abs(flights.loc[f1,'landing time R18R'] - flights.loc[f2,'landing time R18R']) < T_sep:
                AC_sep_LHS = x[f1,0,0] + x[f2,0,0]
                model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1, name=f'AC_sep_f{f2}_f{f1}_r{0}')

            if abs(flights.loc[f1,'landing time R06E'] - flights.loc[f2,'landing time R06E']) < T_sep:
                AC_sep_LHS = x[f1,1,0] + x[f2,1,0]
                model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1, name=f'AC_sep_f{f2}_f{f1}_r{1}')

            if abs(flights.loc[f1,'landing time R24W'] - flights.loc[f2,'landing time R24W']) < T_sep:
                AC_sep_LHS = x[f1,2,0] + x[f2,2,0]
                model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1, name=f'AC_sep_f{f2}_f{f1}_r{2}')

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

model.update()
model.setObjective(obj_func, GRB.MINIMIZE)

model.update()
model.write('model_formulation.lp')

model.optimize()
model.write('solution.sol')

#If the model is feasible, no IIS can be computed
try:
    model.computeIIS()
except:
    pass

model.update()
model.write('model_delete_var.lp')
model.optimize()

try:
    model.computeIIS()
    model.write('model.ilp')
except:
    pass

endTime = time.time()

total_time = endTime - startTime
print('Finished in', round(total_time,2) , 'seconds!')

# Obtaining the results from the model --------------------------------------------------------

with open('solution.sol', newline='\n') as csvfile:
    reader = csv.reader((line.replace(' ', ' ') for line in csvfile), delimiter=' ')
    next(reader)  # skip header
    next(reader)
    sol = {}
    for var, value in reader:
        sol[var] = float(value)

# df_sol = pd.DataFrame.from_records(sol)