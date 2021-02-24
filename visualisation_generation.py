"""
The visualisation code lets the user select one of the generated CSV's and will output various visualisations.

Interesting visualizations:
    > ???
"""

import os

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

path = os.getcwd() + '/schedule_CSV'
files = os.listdir(path)
print(f'List of files ready for visualisation:\n {files}\n')
filename = str(input('Copy + paste file from the list above for visualization'))
df_schedule = pd.read_csv(os.path.join(path, filename))
df_schedule.set_index('Unnamed: 0', inplace=True)

if len(df_schedule) <= 60:
    fs = (30, 10)
else:
    fs = (60, 20)

IAFs, time_to_r, time_to_taxi = gp.multidict({
    'ARTIP': [[1177, 503, 766], [60, 39, 154]],
    'RIVER': [[636, 1177, 1261], [60, 39, 154]],
    'SUGOL': [[709, 1040, 777], [60, 39, 154]]
})

runways = ['R06E', 'R24W', 'R18R']

timeRunwayOccupied = []

# Adding extra column to show the time an aircraft is on the runway.
for i in range(len(df_schedule['Actual Arrival Time at Runway'])):
    runwayIndex = runways.index(df_schedule['Landing Runway'][i])
    IAF = df_schedule['IAF'][i]
    arrTime = df_schedule['Actual Arrival Time at Runway'][i]
    depTimeRW = arrTime + time_to_taxi[IAF][runwayIndex]

    timeRunwayOccupied.append(depTimeRW)

df_schedule['Departure Time Runway'] = timeRunwayOccupied

df_schedule_R18R = df_schedule[df_schedule['Landing Runway'] == 'R18R'].sort_values('Actual Arrival Time at Runway',
                                                                                    ascending=True)
df_schedule_R06E = df_schedule[df_schedule['Landing Runway'] == 'R06E'].sort_values('Actual Arrival Time at Runway',
                                                                                    ascending=True)
df_schedule_R24W = df_schedule[df_schedule['Landing Runway'] == 'R24W'].sort_values('Actual Arrival Time at Runway',
                                                                                    ascending=True)
df_cat_RW = df_schedule.groupby(['Landing Runway', 'Category']).size()

def plotHists():
    plt.figure()
    df_schedule['Delay'].hist()
    plt.title('Histogram of delays')
    plt.xlabel('Delay [s]')
    plt.show()

    plt.figure()
    df_schedule['Landing Runway'].hist()
    df_schedule['IAF'].hist()
    plt.title('Histogram of runway usage vs IAF')
    plt.show()

    plt.figure(figsize=(15, 15))
    sns.pairplot(df_schedule)
    plt.show()
    
    plt.figure()
    df_cat_RW.unstack().plot(kind='bar')
    plt.show()


def plotSchedulesRW():
    subaxis = 0

    if len(df_schedule_R06E) >= 1:
        subaxis += 1
    if len(df_schedule_R24W) >= 1:
        subaxis += 1
    if len(df_schedule_R18R) >= 1:
        subaxis += 1

    fig, ax = plt.subplots(subaxis, figsize=fs, sharex=True)

    if len(df_schedule_R06E) >= 1:
        begin = np.array(df_schedule_R06E['Actual Arrival Time at Runway'].to_list())
        end = np.array(df_schedule_R06E['Departure Time Runway'].to_list())
        ax[0].barh(range(len(begin)), end - begin, left=begin, color='coral')
        ax[0].set_ylabel('Landings at R06R')

    if len(df_schedule_R24W) >= 1:
        begin = np.array(df_schedule_R24W['Actual Arrival Time at Runway'].to_list())
        end = np.array(df_schedule_R24W['Departure Time Runway'].to_list())
        ax[1].barh(range(len(begin)), end - begin, left=begin, color='powderblue')
        ax[1].set_ylabel('Landings at R24R')

    if len(df_schedule_R18R) >= 1:
        begin = np.array(df_schedule_R18R['Actual Arrival Time at Runway'].to_list())
        end = np.array(df_schedule_R18R['Departure Time Runway'].to_list())
        ax[2].barh(range(len(begin)), end - begin, left=begin, color='mediumseagreen')
        ax[2].set_ylabel('Landings at R18R')

    plt.show()


plotHists()
plotSchedulesRW()
