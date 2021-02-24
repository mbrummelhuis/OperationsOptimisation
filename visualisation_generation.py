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

path = os.getcwd() + '/Data_CSVs'
files = os.listdir(path)
print(f'List of files ready for visualisation:\n {files}\n')
filename = str(input('Copy + paste file from the list above for visualization'))
df_data = pd.read_csv(os.path.join(path, filename))
df_data.set_index('Unnamed: 0', inplace=True)

if len(df_data) <= 60:
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
for i in range(len(df_data['Actual Arrival Time at Runway'])):
    runwayIndex = runways.index(df_data['Landing Runway'][i])
    IAF = df_data['IAF'][i]
    arrTime = df_data['Actual Arrival Time at Runway'][i]
    depTimeRW = arrTime + time_to_taxi[IAF][runwayIndex]

    timeRunwayOccupied.append(depTimeRW)

df_data['Departure Time Runway'] = timeRunwayOccupied

windDivision = ['0 - 90', '90 - 180', '180 - 240', '240 - 360']

for i in range(len(df_data)):
    currentWD = df_data.loc[i]['Wind Direction']

    for bounds in windDivision:
        lb = int(bounds.split('-')[0])
        ub = int(bounds.split('-')[-1])

        if lb < currentWD <= ub:
            df_data.loc[i, 'Wind Direction'] = bounds


df_schedule_R18R = df_data[df_data['Landing Runway'] == 'R18R'].sort_values('Actual Arrival Time at Runway',
                                                                            ascending=True)
df_schedule_R06E = df_data[df_data['Landing Runway'] == 'R06E'].sort_values('Actual Arrival Time at Runway',
                                                                            ascending=True)
df_schedule_R24W = df_data[df_data['Landing Runway'] == 'R24W'].sort_values('Actual Arrival Time at Runway',
                                                                            ascending=True)

def plotHists():
    plt.figure()
    df_data['Delay'].hist()
    plt.title('Histogram of delays')
    plt.xlabel('Delay [s]')
    plt.show()

    plt.figure()
    df_data['Landing Runway'].hist()
    df_data['IAF'].hist()
    plt.xlabel('# of flights from IAF and # of flights assigned to runway.')
    plt.show()

    plt.figure(figsize=(40, 40))
    sns.pairplot(df_data)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    df_data.groupby(['Landing Runway', 'Category']).size().unstack().plot(kind='bar')
    plt.show()

    plt.figure(figsize=(10, 10))
    df_data.groupby(['IAF', 'Landing Runway']).size().unstack().plot(kind='bar')
    plt.show()

    plt.figure(figsize=(10, 10))
    df_data.groupby(['Landing Runway', 'Wind Direction']).size().unstack().plot(kind='bar')
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

    if subaxis > 1:
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
    else:
        if len(df_schedule_R06E) >= 1:
            begin = np.array(df_schedule_R06E['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_schedule_R06E['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='coral')
            ax.set_ylabel('Landings at R06R')

        if len(df_schedule_R24W) >= 1:
            begin = np.array(df_schedule_R24W['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_schedule_R24W['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='powderblue')
            ax.set_ylabel('Landings at R24R')

        if len(df_schedule_R18R) >= 1:
            begin = np.array(df_schedule_R18R['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_schedule_R18R['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='mediumseagreen')
            ax.set_ylabel('Landings at R18R')

    plt.show()


plotHists()
plotSchedulesRW()
