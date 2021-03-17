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
savePath = os.getcwd() + '/Visualisations'
files = os.listdir(path)
print(f'List of files ready for visualisation:\n {files}\n')
filename = str(input('Copy + paste file from the list above for visualization'))
save_files = int(input('Do you want to save the files? 1 = yes, 0 = no.'))
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

    occupationTime = time_to_taxi[IAF][runwayIndex]
    # if occupationTime > 120: occupationTime = 120
    depTimeRW = arrTime + occupationTime

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

# Create dataframes per runway
df_data_R18R = df_data[df_data['Landing Runway'] == 'R18R']
df_data_R06E = df_data[df_data['Landing Runway'] == 'R06E']
df_data_R24W = df_data[df_data['Landing Runway'] == 'R24W']

# Create dataframes for costs
df_cost = df_data.groupby('Landing Runway')[['FuelCost', 'NoiseCost', 'WindCost', 'DelayCost']].sum()
df_CoF = df_data.groupby('Landing Runway')[['Flight']].count()
df_cost_norm = df_cost.divide(df_CoF['Flight'].to_list(), axis=0)


def plotHists():
    plt.figure()
    df_data['Delay'].hist()
    plt.title('Histogram of delays')
    plt.xlabel('Delay [s]')
    name = '_Delay_Hist'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()

    # plt.figure()
    # df_data['Landing Runway'].hist()
    # df_data['IAF'].hist()
    # plt.xlabel('# of flights from IAF and # of flights assigned to runway.')
    # plt.show()
    #
    # name = 'Delay_Hist_'
    # savePathFile = os.path.join(save_path, name + filename)
    # if save_files == 1: plt.savefig(savePathFile, dpi='200', format='png')

    plt.figure(figsize=(40, 40))
    sns.pairplot(df_data)
    plt.show()

    plt.figure(figsize=(10, 10))
    df_data.groupby(['Landing Runway', 'Category']).size().unstack().plot(kind='bar')
    name = '_RWxCAT_Hist'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()

    plt.figure(figsize=(10, 10))
    df_data.groupby(['IAF', 'Landing Runway']).size().unstack().plot(kind='bar')
    name = '_IAFxRW_Hist'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()

    plt.figure(figsize=(10, 10))
    df_data.groupby(['Landing Runway', 'Wind Direction']).size().unstack().plot(kind='bar')
    name = '_RWxWD_Hist'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()

def plotCosts():

    plt.figure(figsize=(10, 10))
    df_cost_norm.plot(kind='bar')
    name = '_Cost_Hist'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()

    plt.figure(figsize=(10, 10))
    df_cost.plot(kind='bar')
    name = '_Cost_Hist_CoF'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()




def plotSchedulesRW():
    subaxis = 0

    if len(df_data_R06E) >= 1:
        subaxis += 1
    if len(df_data_R24W) >= 1:
        subaxis += 1
    if len(df_data_R18R) >= 1:
        subaxis += 1

    fig, ax = plt.subplots(subaxis, figsize=fs, sharex=True)

    if subaxis > 1:
        if len(df_data_R06E) >= 1:
            begin = np.array(df_data_R06E['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R06E['Departure Time Runway'].to_list())
            ax[0].barh(range(len(begin)), end - begin, left=begin, color='coral')
            ax[0].set_ylabel('Landings at R06R')

        if len(df_data_R24W) >= 1:
            begin = np.array(df_data_R24W['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R24W['Departure Time Runway'].to_list())
            ax[1].barh(range(len(begin)), end - begin, left=begin, color='powderblue')
            ax[1].set_ylabel('Landings at R24R')

        if len(df_data_R18R) >= 1:
            begin = np.array(df_data_R18R['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R18R['Departure Time Runway'].to_list())
            ax[2].barh(range(len(begin)), end - begin, left=begin, color='mediumseagreen')
            ax[2].set_ylabel('Landings at R18R')
    else:
        if len(df_data_R06E) >= 1:
            begin = np.array(df_data_R06E['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R06E['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='coral')
            ax.set_ylabel('Landings at R06R')

        if len(df_data_R24W) >= 1:
            begin = np.array(df_data_R24W['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R24W['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='powderblue')
            ax.set_ylabel('Landings at R24R')

        if len(df_data_R18R) >= 1:
            begin = np.array(df_data_R18R['Actual Arrival Time at Runway'].to_list())
            end = np.array(df_data_R18R['Departure Time Runway'].to_list())
            ax.barh(range(len(begin)), end - begin, left=begin, color='mediumseagreen')
            ax.set_ylabel('Landings at R18R')

    name = '_FlightSchedule'
    savePathFile = os.path.join(savePath, filename + name + '.png')
    if save_files == 1: plt.savefig(savePathFile, format='png')
    plt.show()


plotHists()
plotCosts()
plotSchedulesRW()
