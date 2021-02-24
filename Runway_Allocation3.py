"""
Gurobi algorithm to solve the MILP Runway Allocation problem for the AE4441-16 Operation Optimisation assignment.

It takes takes the generated landing requests as input for the solver

CHANGES 24.02.21
    > Runway allocation now autosaves to a schedule_CSV folder
        * Naming structure: Schedule_F'#flights'x'frequency'
    > Added new script for data visualisation, which lets you choose which data to visualize

CHANGES:
    > Modified datageneration to output IAF directly, took out code for flights_filtered to handle this change in datageneration
    > Added the part where the flights dataframe is enhanced with landing time at specific runway
    > Took out (old) separation constraint, model is feasible without
    > Made Runway_Allocation3 because I wanted to clean up the code but didn't want to throw away all code that wasn't immediately useful
        > Deleted runway compatibility for wind direction
        > Took out noise penalty from the decision variables
        > Deleted wind restriction and noise limit swithing constraint
    > Reprogrammed separation constraint using the additional data in flights df.

NOTES Kars:
    > Got output although not through the conventional way.
    > No matter the size of the dataset and frequency the maximum delay is 60 seconds.
        * Seems odd to me, maybe has to do with the fact that delay is not integrated in the time separation
        >> Should be looked into

MOST RECENT NOTES MARTIJN:
    > Added the delay loops for the separation constraint
    > Changed cost function to match description in the report (go there for explanation on how it works)

TODO:
    > Validate whether the aircraft separation constraint works properly
    > Do some fancy kind of data visualisation
"""

import csv
import os
import re
import time

import gurobipy as gp
import numpy as np
import pandas as pd
import seaborn as sns
from gurobipy import GRB, LinExpr

sns.set_theme()

# Get path to current folder and select the correct file
cwd = os.getcwd()
filename = 'landing_requests.csv'

startTime = time.time()

# Load the flight data
flights = pd.read_csv(os.path.join(cwd, filename), delimiter=',', skiprows=2, header=0)

# Stored constants------------------------------------------------------------------------------------------
runways, runway_headings, runway_pop = gp.multidict({
    'R06E': [60, 700],
    'R24W': [240, 13600],
    'R18R': [180, 14500]
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
    for IAF in range(len(IAFs)):  # Iterate over IAFS
        if IAFs[IAF] == flights['IAF'][f]:
            flights.loc[f, 'landing time R18R'] = time_to_r[IAFs[IAF]][0] + flights['time in seconds'][f]
            flights.loc[f, 'landing time R06E'] = time_to_r[IAFs[IAF]][1] + flights['time in seconds'][f]
            flights.loc[f, 'landing time R24W'] = time_to_r[IAFs[IAF]][2] + flights['time in seconds'][f]

# Print flights for checking      
print(flights)

# Create a dataframe which includes the cost per arc (CPA) from IAF to runway
cpa = []
for AC in AC_type:
    for n in range(len(IAFs)):
        for r in range(len(runways)):
            cpa.append([IAFs[n], runways[r], AC, (time_to_r[IAFs[n]][r] * AC_fuel[AC] / 1297.569),
                        runway_pop[runways[r]] / 14500])

df_cpa = pd.DataFrame(cpa)
df_cpa.columns = ['IAF', 'Runway', 'AC', 'Fuel cost', 'Noise cost']
print(df_cpa)

'''
Setting up the decision variables:
    > flight_{f, r, d}: arriving flight from IAF f to runway r, with delay d
'''
x = {}

model = gp.Model('runway_allocation')  # Initiate model

dt = 60  # delta time delay in seconds
delay_steps = 10  # maximum number of delay steps (minutes)
delays = np.ones(delay_steps).tolist()
delays[0] = 0
for i in range(len(delays) - 1):
    delays[i + 1] = delays[i] + dt

for f in range(len(flights['IAF'])):
    for r in range(len(runways)):
        for d in delays:
            x[f, r, d] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                      name='x%s_%s_%s' % (f, r, d))  # name='x%s_%s_%s' % (f, r, d)

model.update()

'''
Adding the constraints:
    > Always Assign Flight: make sure that every AC gets assigned
    > AC Landing sep: make sure that there is enough time between consecutive arrivals
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
        for d1 in delays:
            for d2 in delays:
                if f1 > f2:  # > so flights are only compared once
                    if abs((flights.loc[f1, 'landing time R18R'] + d1) - (
                            flights.loc[f2, 'landing time R18R'] + d2)) < T_sep:
                        AC_sep_LHS = x[f1, 0, d1] + x[f2, 0, d2]
                        model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1,
                                        name=f'AC_sep_f{f2}_f{f1}_r{0}_delay{d1}/{d2}')

                    if abs((flights.loc[f1, 'landing time R06E'] + d1) - (
                            flights.loc[f2, 'landing time R06E'] + d2)) < T_sep:
                        AC_sep_LHS = x[f1, 1, d1] + x[f2, 1, d2]
                        model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1,
                                        name=f'AC_sep_f{f2}_f{f1}_r{1}_delay{d1}/{d2}')

                    if abs((flights.loc[f1, 'landing time R24W'] + d1) - (
                            flights.loc[f2, 'landing time R24W'] + d2)) < T_sep:
                        AC_sep_LHS = x[f1, 2, d1] + x[f2, 2, d2]
                        model.addConstr(lhs=AC_sep_LHS, sense=GRB.LESS_EQUAL, rhs=1,
                                        name=f'AC_sep_f{f2}_f{f1}_r{2}_delay{d1}/{d2}')
                    else:
                        pass

model.update()

'''
Generating the cost function
 > Minimize the cost
 > Cost increases as delay increases
'''
obj_func = LinExpr()

for f in range(len(flights['IAF'])):
    for r in range(len(runways)):
        for d in delays:
            fuel_cost = df_cpa.loc[(df_cpa['IAF'] == flights['IAF'][f]) &
                                   (df_cpa['AC'] == flights['category'][f]) &
                                   (df_cpa['Runway'] == runways[r]), 'Fuel cost'].values[0]
            noise_cost = df_cpa.loc[(df_cpa['IAF'] == flights['IAF'][f]) &
                                    (df_cpa['AC'] == flights['category'][f]) &
                                    (df_cpa['Runway'] == runways[r]), 'Noise cost'].values[0]
            if abs(runway_headings[runways[r]] - flights['wind direction'][f]) > 180:
                wind_cost = (360 - abs(runway_headings[runways[r]] - flights['wind direction'][f])) / 180
            else:
                wind_cost = abs(runway_headings[runways[r]] - flights['wind direction'][f]) / 180
            delay_cost = d / 600
            C_F = fuel_cost + noise_cost + wind_cost + delay_cost  # Add all the costs
            obj_func += C_F * x[f, r, d]  # Add coefficient multiplied by DV to cost function

model.update()
model.setObjective(obj_func, GRB.MINIMIZE)

model.update()
model.write('model_formulation.lp')

model.optimize()
model.write('solution.sol')

# If the model is feasible, no IIS can be computed
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
print('Finished in', round(total_time, 2), 'seconds!')

# Obtaining the results from the model --------------------------------------------------------

with open('solution.sol', newline='\n') as csvfile:
    reader = csv.reader((line.replace(' ', ' ') for line in csvfile), delimiter=' ')
    next(reader)  # skip header
    next(reader)
    landedFlights = []
    used_runways = []
    flight_delay = []
    arrTimeRunway = []
    ActArrTimeRunway = []

    for var, value in reader:
        value = float(value)
        if value == 1:  # Only account for the assigned decision variables
            varSplit = var.split('_')  # Splitting the name of the variable: [flight, runway, delay]

            # Index of the flight, to obtain the arrival time
            flightIndex = int(re.compile('([a-zA-Z]+)([0-9]+)').match(varSplit[0]).groups()[1])
            runwayIndex = int(varSplit[1])
            flightDelay = float(varSplit[2])

            # Storing the flight schedule for all flights
            landedFlights.append(varSplit[0])
            used_runways.append(runways[runwayIndex])
            flight_delay.append(flightDelay)

            # Storing the landing time on runway
            landing_str = 'landing time ' + runways[runwayIndex]
            arrTimeRunway.append(flights[landing_str][flightIndex])
            ActArrTimeRunway.append(flights[landing_str][flightIndex] + flightDelay)

df_schedule = pd.DataFrame(
    list(zip(landedFlights, flights['IAF'].to_list(), flights['category'].to_list(), flight_delay, used_runways, arrTimeRunway, ActArrTimeRunway)),
    columns=['Flight', 'IAF', 'Category', 'Delay', 'Landing Runway', 'Planned Arrival Time at Runway',
             'Actual Arrival Time at Runway'])

# Saving the simulation data in a CSV.
frequency = str(int(len(df_schedule) / (flights['time in seconds'].to_list()[-1] / 3600)))
scheduleName = 'Schedule_F' + str(len(df_schedule)) + f'x{frequency}'  # number of flights x frequency
savePath = os.getcwd() + '/Schedule_CSV'
df_schedule.to_csv(os.path.join(savePath, scheduleName))
