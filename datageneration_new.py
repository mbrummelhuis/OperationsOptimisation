"""
Data generation algorithm for AE4441-16 Operations Optimisation assignment
Generating random input data for the optimisation algorithm

Creates a .csv file in the file directory with the specified number of data points
The user inputs to this are:
    Amount of data points
    The flight frequency in average flights per hour

Each data point represents an aircraft requesting landing at an arbitrary airport.
For each aircraft, the following data is available:
    Time of coming in (requesting the landing at tower)
    Aircraft type (class based on weight)
    Include direction of approach
    
    - Calculate time to sensible format
    - for some reason it skips each other row, fix this
"""
import csv
import random
import numpy as np

def convertTime(seconds): 
    day = seconds // (24 * 3600)
    seconds %= (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    time = str(day) + ":" + format(hour, '02d') + ":" + format(minutes, '02d') + ":" + format(seconds, '02d')
      
    return time

time_seconds = 0
headings = np.arange(0,360,1)


#Division between Medium and Heavy aircraft (sum to 1)
medium = 0.7
heavy = 0.3

print("Data generation for AE4441-16 Operations optimisation.")
flights = int(input("Please enter the amount of data points (flights): "))
flight_frequency = input("Please enter the average flight frequency (flights/hr): ")

mean_time_between_flights = float(3600)/float(flight_frequency)
stdev_time_between_flights = float(mean_time_between_flights)/float(4)

#Initializing csv file
csv_file = open('landing_requests.csv', 'w')

with open('landing_requests.csv','w') as data_file:
    writer = csv.writer(data_file,lineterminator = '\n')
    
    writer.writerow(['amount of flights',flights])
    writer.writerow(['average flights per hour', flight_frequency])

    writer.writerow(['time in seconds', 'time in d:hh:mm:ss', 'category', 'approach direction', 'wind direction'])
    for flight in range(flights):
        time_since_last_flight = np.random.normal(mean_time_between_flights, stdev_time_between_flights)
        time_seconds = int(time_seconds + time_since_last_flight)
        
        category_probability = random.random()
        if category_probability < medium:
            category = 'Medium'
        else:
            category = 'Heavy'

        approach_direction = random.choice(headings)
        
        wind_direction = random.choice(headings)
        
        time_hhmmss = convertTime(time_seconds)
        
        writer.writerow([time_seconds, time_hhmmss, category, approach_direction, wind_direction])

print('Done')