# All Python Lecture Notes
""" Collection of all my python lectures"""

Introduction to Python

"""
pthon basics
python lists
functions and packages
statistics
numpy
"""

# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]
# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]
# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Create areas_copy
areas_copy = areas[:]
# Change areas_copy
areas_copy[0] = 5.0
# Print areas
print(areas)


# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True
# Print out type of var1
print(type(var1))
# Print out length of var1
print(len(var1))
# Convert var2 to an integer: out2
out2 = int(var2)


# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]
# Paste together first and second: full
full = first + second
# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse = True)
# Print out full_sorted
print(full_sorted)


# string to experiment with: place
place = "poolhouse"
# Use upper() on place: place_up
place_up = place.upper()
# Print out place and place_up
print(place, place_up)
# Print out the number of o's in place
print(place.count("o"))


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Print out the index of the element 20.0
print(areas.index(20.0))
# Print out how often 9.50 appears in areas
print(areas.count(9.50))


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)
# Print out areas
print(areas)
# Reverse the orders of the elements in areas
areas.reverse()
# Print out areas
print(areas)


# Definition of radius
r = 0.43
# Import the math package
import math
# Calculate C
C = 2 * math.pi * r
# Calculate A
A = math.pi * r ** 2
# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))


# Definition of radius
r = 192500
# Import radians function of math package
from math import radians
# Travel distance of Moon over 12 degrees. Store in dist.
phi = radians(12)
dist = phi * r
# Print out dist
print(dist)


# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]
# Import the numpy package as np
import numpy as np
# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)
# Print out type of np_baseball
print(type(np_baseball))


# height_in is available as a regular list
# Import numpy
import numpy as np
# Create a numpy array from height_in: np_height_in
np_height_in = np.array(height_in)
# Print out np_height_in
print(np_height_in)
# Convert np_height_in to m: np_height_m
np_height_m = np_height_in * 0.0254
# Print np_height_m
print(np_height_m)


# height_in and weight_lb are available as regular lists
# Import numpy
import numpy as np
# Create array from height_in with metric units: np_height_m
np_height_m = np.array(height_in) * 0.0254
# Create array from weight_lb with metric units: np_weight_kg
np_weight_kg = np.array(weight_lb) * .453592
# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m ** 2
# Print out bmi
print(bmi)


# height_in and weight_lb are available as a regular lists
# Import numpy
import numpy as np
# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2
# Create the light array
light = bmi < 21
# Print out light
print(light)
# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])


# height_in and weight_lb are available as a regular lists
# Import numpy
import numpy as np
# Store weight and height lists as numpy arrays
np_weight_lb = np.array(weight_lb)
np_height_in = np.array(height_in)
# Print out the weight at index 50
print(np_weight_lb[50])
# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])


# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
# Import numpy
import numpy as np
# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)
# Print out the type of np_baseball
print(type(np_baseball))
# Print out the shape of np_baseball
print(np_baseball.shape)


# baseball is available as a regular list of lists
# Import numpy package
import numpy as np
# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)
# Print out the shape of np_baseball
print(np_baseball.shape)


# baseball is available as a regular list of lists
# Import numpy package
import numpy as np
# Create np_baseball (2 cols)
np_baseball = np.array(baseball)
# Print out the 50th row of np_baseball
print(np_baseball[49, : ])
# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:,1]
# Print out height of 124th player
print(np_baseball[123,0])


# baseball is available as a regular list of lists
# updated is available as 2D numpy array
# Import numpy package
import numpy as np
# Create np_baseball (3 cols)
np_baseball = np.array(baseball)
# Print out addition of np_baseball and updated
print(np_baseball + updated)
# Create numpy array: conversion
conversion = np.array([0.0254, .453592, 1])
# Print out product of np_baseball and conversion
print(np_baseball * conversion)


""" Statistics"""
# np_baseball is available
# Import numpy
import numpy as np
# Create np_height_in from np_baseball
np_height_in = np_baseball[:,0]
# Print out the mean of np_height_in
print(np.mean(np_height_in))
# Print out the median of np_height_in
print(np.median(np_height_in))


# np_baseball is available
# Import numpy
import numpy as np
# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))
# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))
# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))
# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0], np_baseball[:,1])
print("Correlation: " + str(corr))


# heights and positions are available as lists
# Import numpy
import numpy as np
# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions = np.array(positions)
np_heights = np.array(heights)
# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK']
# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']
# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))
# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))


Intermediate Python
"""
Matplot lib
Dictionaries
Pandas
loops
random numbers
"""


# Matplot Lib

"""Line Plots"""
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()


"""Scatter Plots"""
# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)
# Put the x-axis on a logarithmic scale
plt.xscale('log')
# Show plot
plt.show()


# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 
# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'
# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)
# Add title
plt.title(title)
# After customizing, display the plot
plt.show()

# Scatter plot
plt.scatter(gdp_cap, life_exp)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']
# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)
# After customizing, display the plot
plt.show()

# Super Beefy Scatter Plot
# Import numpy as np
import numpy as np
# Store pop as a numpy array: np_pop
np_pop = np.array(pop)
# Double np_pop
np_pop = 2 * np_pop
# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
# Display the plot
plt.show()


"""histograms"""
# Build histogram with 5 bins
plt.hist(life_exp, 5)
# Show and clean up plot
plt.show()
plt.clf()
# Build histogram with 20 bins
plt.hist(life_exp, 20)
# Show and clean up again
plt.show()
plt.clf()


"""Dictionaries"""

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
# Get index of 'germany': ind_ger
ind_ger = countries.index("germany")
# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
# From string in countries and capitals, create dictionary europe
europe = {'spain':'madrid', 'france': 'paris', 'germany': 'berlin',  'norway': 'oslo'}
# Print europe
print(europe)


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Print out the keys in europe
print(europe.keys())
# Print out value that belongs to key 'norway'
print(europe['norway'])


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' } 
# Add italy to europe
europe['italy'] = 'rome'
# Print out italy in europe
print('italy' in europe)
# Add poland to europe
europe['poland'] = 'warsaw'
# Print europe
print(europe)


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }
# Update capital of germany
europe['germany'] = 'berlin'
# Remove australia
del(europe['australia'])
# Print europe
print(europe)


# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
# Print out the capital of France
print(europe['france']['capital'])
# Create sub-dictionary data
data = {'capital': 'rome', 'population': 59.83}
# Add data to europe under key 'italy'
europe['italy'] = data
# Print europe
print(europe)


"""Pandas"""
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
# Import pandas as pd
import pandas as pd
# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country': names, 'drives_right': dr, 'cars_per_cap': cpc}
# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)
# Print cars
print(cars)


import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)
# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
# Specify row labels of cars
cars.index = row_labels
# Print cars again
print(cars)


# Import pandas as pd
import pandas as pd
# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')
# Print out cars
print(cars)


# Import pandas as pd
import pandas as pd
# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out cars
print(cars)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out country column as Pandas Series
print(cars['country'])
# Print out country column as Pandas DataFrame
print(cars[['country']])
# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
print(cars)
# Print out first 3 observations
print(cars[0:3])
# Print out fourth, fifth and sixth observation
print(cars[3:6])


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out observation for Japan
print(cars.loc['JPN'])
# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out drives_right value of Morocco
print(cars.loc['MOR', 'drives_right'])
# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out drives_right column as Series
print(cars.loc[:,'drives_right'])
# Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])
# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap','drives_right']])


# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than or equal to 18
print(my_house >= 18)
# my_house less than your_house
print(my_house < your_house)


# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, your_house <10))
# Both my_house and your_house smaller than 11
print(np.logical_and(my_house< 11, your_house < 11))


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Extract drives_right column as Series: dr
dr = cars.loc[:,'drives_right']
# Use dr to subset cars: sel
sel = cars[dr]
# Print sel
print(sel)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Convert code to a one-liner
sel = cars[cars['drives_right']]
# Print sel
print(sel)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars.loc[:,'cars_per_cap']
#print('cpc', cpc)
many_cars = cpc > 500
#print(many_cars)
# Print car_maniac
car_maniac = cars[many_cars]
print(car_maniac)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Import numpy, you'll need this
import numpy as np
# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]
# Print medium
print(medium)


# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index + 1) + ": " + str(area))


# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
# Build a for loop from scratch
for room_number, room in enumerate(house):
    print("the " + room[0] + " is " + str(room[1]) + " sqm")


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
# Iterate over europe
for country, capital in europe.items():
    print("the capital of " + country + " is " + capital)


# Import numpy as np
import numpy as np
# For loop over np_height
for height in np_height:
    print(str(height) + " inches")
# For loop over np_baseball
#to loop over ALL elements in a 2D array
for ele in np.nditer(np_baseball):
    print(ele, end = ",")


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab +": " + str(row['cars_per_cap']))


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, 'COUNTRY'] = row['country'].upper()
# Print cars
print(cars)


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
cars['COUNTRY'] = cars['country'].apply(str.upper)
print(cars)


"""random numbers"""
# Import numpy as np
import numpy as np
# Set the seed
np.random.seed(123)
# Generate and print random float
print(np.random.rand())


# Import numpy and set seed
import numpy as np
np.random.seed(123)
# Use randint() to simulate a dice
print(np.random.randint(1,7))
# Use randint() again
print(np.random.randint(1,7))


# NumPy is imported, seed is set

# Starting step
step = 50
# Roll the dice
dice = np.random.randint(1,7)
# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <=5 :
    step += 1
else :
    step = step + np.random.randint(1,7)
# Print out dice and step
print(dice, step)


# NumPy is imported, seed is set
# Initialize random_walk
random_walk = [0]
# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]
    # Roll the dice
    dice = np.random.randint(1,7)
    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    # append next_step to random_walk
    random_walk.append(step)
# Print random_walk
print(random_walk)


# NumPy is imported, seed is set
# Initialize random_walk
random_walk = [0]
for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)
    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = step - 1
        step = max( 0 , step)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step)
print(random_walk)


# NumPy is imported, seed is set
# Initialization
random_walk = [0]
for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step)
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Plot random_walk
plt.plot(random_walk)
# Show the plot
plt.show()


# NumPy is imported; seed is set
# Initialize all_walks (don't change this line)
all_walks = []
# Simulate random walk 10 times
for i in range(10) :
    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    # Append random_walk to all_walks
    all_walks.append(random_walk)
# Print all_walks
print(all_walks)


# numpy and matplotlib imported, seed set.
# initialize and populate all_walks
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)
# Convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)
# Plot np_aw and show
plt.plot(np_aw)
plt.show()
# Clear the figure
plt.clf()
# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)
# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


# numpy and matplotlib imported, seed set
# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= .001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)
# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()



# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)
# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
# Select last row from np_aw_t: ends
ends = np_aw_t[-1, :]
# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()


# Python Notes
"""
dictionaries
csvs
collections
default dicts
named touple
datetime
timezone


"""
for index, item in enumerate(list_of_things):
	print(index, item)

#sets
# use set() to create a set
.union()
.intersection()
.difference()


#Dictionaries
.get('key', "Message if it's not found") #is a method that will grab a dictionary entry if it exists
print(names.get(105, "Not Found"))
sorted(names_by_rank, reverse = True)[:10] #prints in decending order
#Keys to a dictionary
print(boy_names.keys())
print(boy_names[2013].keys())
#updating dictionaries
# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011
# Update the 2012 entry in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])
# Loop over the years in the boy_names dictionary 
for year in boy_names:
    # Sort the data for each year by descending rank and get the lowest one
    lowest_ranked =  sorted(boy_names[year], reverse=True)[0]
    # Safely print the year and the least popular name or 'Not Available'
    print(year, boy_names[year].get(lowest_ranked, 'Not Available')  )



# Remove 2011 from female_names and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 from female_names with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015, {})

# Delete 2012 from female_names
del female_names[2012]

# Iterate over the 2014 nested dictionary
for rank, names in baby_names[2014].items():
    # Print rank and name
    print(rank, names)


# WORKING WITH CSVs


# Use [1:] to skip the header row
# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())

# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())


# COLLECTIONS MODULE
.most_common()
# Import the Counter object
from collections import Counter

# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Find the 5 most common elements
print(station_count.most_common(5))

# Print the station_count
print(station_count)


# Default Dict
from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

for eatery in nyc_eateries:
	if eatery.get('phone'):
		eatery_contact_types['phone'] += 1
	if eatery.get('website'):
		eatery_contact_types['websites'] += 1

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the ridership dictionary
    if date not in ridership:
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))
    
# Print the ridership for '03/09/2016'
print(ridership['03/09/2016'])

## 
from collections import defaultdict

# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    ridership[stop].append(riders)
    
# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:10])

from collections import OrderedDict
OrderedDict()
nyc_eatery_permits = OrderedDict()
for eatery in nyc_eateries:
	nyc_eatery_permits[eater['end_date']] = eatery
print(list(nyc_eatery_permits.items())[:3])

nyc_eatery_permits.popitem()
nyc_eatery_permits.popitem(last = False) #keyword argument to return items in insertion order

# Named Tuple
from collections import namedtouple
Eatery = namedtouple('Eatery', ['name', 'location', 'park_id','type_name'])
for eatery in nyc_eatery:
	details = Eatery(eatery['name'],
					eatery['location'],
					eatery['park_id'],
					eatery['type_name'])
	eateries.append(details)
print(Eateries[0])

for eatery in eateries[:3]:
	print(eatery.name)
	print(eatery.park_id)
	print(eatery.location)


# Import namedtuple from collections
from collections import namedtuple

# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries list
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))
    
# Print the first 5 items in labeled_entries
print(labeled_entries[:5])

# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop
    print(item.stop)
    # Print each item's date
    print(item.date)
    # Print each item's riders
    print(item.riders)

# DATES AND TIMES
import datetime
.strptime() method convers a string to datetime
date_dt = datetime.strptime(parking_violations_date, '%m/%d/%Y')
.strftime() method converts a string to a datetime
date_dt.strftime('%m/%d/%Y')
isoformat() 

daily_violations = defaultdict(int)
for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m,%d,%Y')
	parking_violations[violation_date.day] += 1
print(sorted(daily_violations.items()))


.now() #returns local datetime on your machine 
.utcnow() #returns the current UTC datetime

now_dt = datetime.now()
utc_dt = datetime.utcnow()

#timezones!
#Naive datetime objects have no timezone data
# Aware datetime objects have a timezone.
# Timezone data is available viq the pytz module via the timezone object
# aware objects have .astimezone() so you can get the time in another time zone.
from pytz import timezone
record_dt = datetime.strptime('07/12/2016 04:39PM', '%m/%d/%Y')
ny_tz = timezone('US/Eastern')
a_tz = timezone('US/Pacific')
ny_dt = record_dt.replace(tzinfor=ny_tz) #makes the naive datetime into aware.
la_dt = ny_dt.astimezone(la_tz)


# adding and subtracting time
timedelta # is used to represent an amount of change in time.
from datetime import timedelta
flashback = timedelta(days = 90)
print(record_dt)
print(record_dt - flashback)
print(record_dt + flashback)

time_diff = record_dt - record2_dt
type(time_diff)
# datetime.timedelta

# Help! Libraries
.parse() will attempt to conver a string to a pendulum datetime object
import pedulum
occured = violation[4] + ' ' + violation[5] + 'M'
occured_dt = predulum.parse(occurred, tz='US/Easter')
print(occured_dt)

.in_timezone() method converts a pendulum time object to a desired timezone.
print(pendulum.now('Asia/Tokyo'))


.in_XXX() # Method procide the differences in a chosen metric
.in_words() # method provides the difference in a nice form

diff = violation_dts[3] - violation_dts[2]
print(diff.in_words())
print(diff.in_days())
print(diff.in_hours())

import pendulum
# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')
# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')=
# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())


#Chicago Data set
Date, Block, Primary, Type, Description, Location Description, Arrest, Domestic, District
05/23/2016 05:35:00 PM, 024XX W DIVISION ST, ASSULT, SIMPLE, STREET, false, true, 14

from collections import Counter

nyc_eatery_count_by_types = Counter(nyc_eatery_types)

daily_violations = defaultdict(int)

for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m/%d/%Y')
	daily_violations[violation_date.day] += 1


from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

print(nyc_eatery_count_by_types.most_common(3))


# PART 2
"""
First determine how many crimes occured by district, and then look at crime by district"""
import csv
csvfile = open('ART_GALLERY.csv', 'r')
for row in csv.DictReader(csvfile):
    print(row)

# pop out a value from the dictionary
galleries_10310 = artgalleries.pop('10310')

# Loop over dictionary
for zip_code, galleries in Art_calleries.items():
    print(zip_code)
    print(galleries)

cookies_eaten_today = ['chocolate chip', 'peanut butter', 'chocolate chip', 'oatmeal cream', 'chocolate chip']
types_of_cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)

difference()
cookies_jason_ate.difference(cookies_hugo_ate)
set(['oatmeal cream', 'peanut butter'])



# Python Notes
"""
dictionaries
csvs
collections
default dicts
named touple
datetime
timezone


"""
for index, item in enumerate(list_of_things):
	print(index, item)

#sets
# use set() to create a set
.union()
.intersection()
.difference()


#Dictionaries
.get('key', "Message if it's not found") #is a method that will grab a dictionary entry if it exists
print(names.get(105, "Not Found"))
sorted(names_by_rank, reverse = True)[:10] #prints in decending order
#Keys to a dictionary
print(boy_names.keys())
print(boy_names[2013].keys())
#updating dictionaries
# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011
# Update the 2012 entry in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])
# Loop over the years in the boy_names dictionary 
for year in boy_names:
    # Sort the data for each year by descending rank and get the lowest one
    lowest_ranked =  sorted(boy_names[year], reverse=True)[0]
    # Safely print the year and the least popular name or 'Not Available'
    print(year, boy_names[year].get(lowest_ranked, 'Not Available')  )



# Remove 2011 from female_names and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 from female_names with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015, {})

# Delete 2012 from female_names
del female_names[2012]

# Iterate over the 2014 nested dictionary
for rank, names in baby_names[2014].items():
    # Print rank and name
    print(rank, names)


# WORKING WITH CSVs


# Use [1:] to skip the header row
# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())

# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())


# COLLECTIONS MODULE
.most_common()
# Import the Counter object
from collections import Counter

# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Find the 5 most common elements
print(station_count.most_common(5))

# Print the station_count
print(station_count)


# Default Dict
from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

for eatery in nyc_eateries:
	if eatery.get('phone'):
		eatery_contact_types['phone'] += 1
	if eatery.get('website'):
		eatery_contact_types['websites'] += 1

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the ridership dictionary
    if date not in ridership:
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))
    
# Print the ridership for '03/09/2016'
print(ridership['03/09/2016'])

## 
from collections import defaultdict

# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    ridership[stop].append(riders)
    
# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:10])

from collections import OrderedDict
OrderedDict()
nyc_eatery_permits = OrderedDict()
for eatery in nyc_eateries:
	nyc_eatery_permits[eater['end_date']] = eatery
print(list(nyc_eatery_permits.items())[:3])

nyc_eatery_permits.popitem()
nyc_eatery_permits.popitem(last = False) #keyword argument to return items in insertion order

# Named Tuple
from collections import namedtouple
Eatery = namedtouple('Eatery', ['name', 'location', 'park_id','type_name'])
for eatery in nyc_eatery:
	details = Eatery(eatery['name'],
					eatery['location'],
					eatery['park_id'],
					eatery['type_name'])
	eateries.append(details)
print(Eateries[0])

for eatery in eateries[:3]:
	print(eatery.name)
	print(eatery.park_id)
	print(eatery.location)


# Import namedtuple from collections
from collections import namedtuple

# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries list
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))
    
# Print the first 5 items in labeled_entries
print(labeled_entries[:5])

# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop
    print(item.stop)
    # Print each item's date
    print(item.date)
    # Print each item's riders
    print(item.riders)

# DATES AND TIMES
import datetime
.strptime() method convers a string to datetime
date_dt = datetime.strptime(parking_violations_date, '%m/%d/%Y')
.strftime() method converts a string to a datetime
date_dt.strftime('%m/%d/%Y')
isoformat() 

daily_violations = defaultdict(int)
for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m,%d,%Y')
	parking_violations[violation_date.day] += 1
print(sorted(daily_violations.items()))


.now() #returns local datetime on your machine 
.utcnow() #returns the current UTC datetime

now_dt = datetime.now()
utc_dt = datetime.utcnow()

#timezones!
#Naive datetime objects have no timezone data
# Aware datetime objects have a timezone.
# Timezone data is available viq the pytz module via the timezone object
# aware objects have .astimezone() so you can get the time in another time zone.
from pytz import timezone
record_dt = datetime.strptime('07/12/2016 04:39PM', '%m/%d/%Y')
ny_tz = timezone('US/Eastern')
a_tz = timezone('US/Pacific')
ny_dt = record_dt.replace(tzinfor=ny_tz) #makes the naive datetime into aware.
la_dt = ny_dt.astimezone(la_tz)


# adding and subtracting time
timedelta # is used to represent an amount of change in time.
from datetime import timedelta
flashback = timedelta(days = 90)
print(record_dt)
print(record_dt - flashback)
print(record_dt + flashback)

time_diff = record_dt - record2_dt
type(time_diff)
# datetime.timedelta

# Help! Libraries
.parse() will attempt to conver a string to a pendulum datetime object
import pedulum
occured = violation[4] + ' ' + violation[5] + 'M'
occured_dt = predulum.parse(occurred, tz='US/Easter')
print(occured_dt)

.in_timezone() method converts a pendulum time object to a desired timezone.
print(pendulum.now('Asia/Tokyo'))


.in_XXX() # Method procide the differences in a chosen metric
.in_words() # method provides the difference in a nice form

diff = violation_dts[3] - violation_dts[2]
print(diff.in_words())
print(diff.in_days())
print(diff.in_hours())

import pendulum
# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')
# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')=
# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())


#Chicago Data set
Date, Block, Primary, Type, Description, Location Description, Arrest, Domestic, District
05/23/2016 05:35:00 PM, 024XX W DIVISION ST, ASSULT, SIMPLE, STREET, false, true, 14

from collections import Counter

nyc_eatery_count_by_types = Counter(nyc_eatery_types)

daily_violations = defaultdict(int)

for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m/%d/%Y')
	daily_violations[violation_date.day] += 1


from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

print(nyc_eatery_count_by_types.most_common(3))


# PART 2
"""
First determine how many crimes occured by district, and then look at crime by district"""
import csv
csvfile = open('ART_GALLERY.csv', 'r')
for row in csv.DictReader(csvfile):
    print(row)

# pop out a value from the dictionary
galleries_10310 = artgalleries.pop('10310')

# Loop over dictionary
for zip_code, galleries in Art_calleries.items():
    print(zip_code)
    print(galleries)

cookies_eaten_today = ['chocolate chip', 'peanut butter', 'chocolate chip', 'oatmeal cream', 'chocolate chip']
types_of_cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)

difference()
cookies_jason_ate.difference(cookies_hugo_ate)
set(['oatmeal cream', 'peanut butter'])




#Data Manipulation with Pandas
"""
1. Tansforming DataFrames
    sorting and subsetting
    creating new columns
2. Aggregating DataFrames
    summary statistics
    counting
    grouped summary statistics
3. Slicing and Indexing DataFrames
    subsetting using slicing
    indexes and subsetting using indexes
4. Creating and Visualizing DataFrames
    plotting
    hand missing data
    reding data into a DataFrame
"""

df.head() # returns the first few rows (the “head” of the DataFrame).
df.info() # shows information on each of the columns, such as the data type and number of missing values.
df.shape # returns the number of rows and columns of the DataFrame.
    # Note .shape doesn't use () because it is an atribute not a method
df.describe() # calculates a few summary statistics for each column.
    # count is the number of non-missing values in a column


#data frames consist of 3 different components
df.values
    # contains the data values in a 2D numpy array
df.columns
    # contains the column labels
df.index        
    # contains row numbers not row names



"Sorting and Subsetting"

# you can arrange data by column.
# if you want the biggest first, ascending is set to True by default.
df.sort_values("column_name", ascending = False)

#You can also pass in a set of columns.
df.sort_values(['weight_kg', 'height_cm'])

# If you would like to sort each column differently pass in an ascending list.
df.sort_values(['weight_kg', 'height_cm'], ascending = [True, False])

#will return a column
df['column_name'] 

# to return multiple columns, insert a list of column names
df[['column_name1', 'column_name2']]

df['height_cm'] > 50 #will return a list of boolean values

#use this list inside to grab only the true rows.
dogs[dogs["height_cm"] > 50]
# OR based on text
dogs[dogs['breed'] == 'Labrador']
# or based on dates
dogs[dogs['date_of_birth'] < "2015-01-01"]

# You can use logical operators as well to combine conditions.
is_lab = dogs['breed'] == 'Labrador'
is_brown = dogs['color'] == 'Brown'
dog[is_lab & is_brown]
dog[ (dogs['breed'] == 'Labrador') & (dogs['color'] == 'Brown') ]

#df.isin() is a way to subset using multiple catagorical columns
is_black_or_brown = dogs['color'].isin(['Black', 'Brown'])
dogs[is_black_or_brown]

"""new columns"""
# adding a new column.
# going from cm to M
dogs['height_m'] = dogs['height_cm'] / 100
print(dogs)

#calculating BMI  weight/height^2
#adding a new BMI column
dogs['bmi'] = dogs['weight_kg'] / dogs['height_m'] ** 2
# print(dogs.head())


"""Summary statistics"""
dogs["height_cm"].mean()
.median()
.mode()
.min()
.max()
.var()
.std()
.sum()
.quantile()
dog['date_of_birth'].min() # find the oldest dog
dog['date_of_birth'].max() # find the youngest dog

#The .agg() method allows you to compute custom summary statistices.

# returns the 30th percentile
def pct30(column):
    return column.quantile(0.3)

dogs['weight_kg'].agg(pct30)

#using .agg() on multiple columns
dogs[['weight_kg', 'height_cm']].agg(pct30)


def pct40(column):
    return column.quantile(0.4)


# You can pass a list of functions into .agg()
dogs["weight_kg"].agg([pct30, pct40])
#returns the 30th percentile and the 40th

"cumulative sum"
.cumsum()
dogs['weight_kg'].cumsum()
index   list    cumsum
0       24      24
1       24      48
2       24      72
3       17      89      
4       29      118
5       2       120
6       74      194


# returns an entire column of data
.cummax() 
.cummin() 
.cumprod() 


store type  department       date  weekly_sales  is_holiday  temperature_c  fuel_price_usd_per_l  unemployment


"""Counting categorical data"""
drop_duplicates() # removing rows with duplicate values in their columns
vet_visits.drop_duplicate(subset='name') #drops all but one record with duplicate names

unique_dogs = vet_visits.drop_duplicates(subset = ['name', 'breed']) #keeps records with a 
# unique name and breed combination.
print(unique_dogs)

unique_dogs['breed'].value_counts() #gets the number of dogs of each breed
unique_dogs['breed'].value_counts(sort=True) # puts the largest values at the top

#normalize can e used to put counts in proportion of the total
unique_dogs['breed'].value_counts(normalize=True)


groupby() #this is a method that will group records by what they have in common

dogs.groupy('color')['weight_kg'].mean() #this will give the mean weight of each dog by color

#Use .agg() to get multiple statistics
dogs.groupby('color')['weight_kg'].agg([min, max, sum]) #will return the min, max, and sum of 
# each group of dogs by their colors.

# you can group by multiple statistics to make a summary
dogs.groupby(['color', 'breed'])['weight_kg'].mean()
# returns each color and then subsetted into dog breeds and the average weight of each dog.


"""pivot tables"""
dogs.groupby('color')['weight_kg'].mean()

#pivot tables allow us to easily calculate summary statistics
# The values column is the one you want to summarize
# The index colum is the one you want to group by
dogs.pivot_table(values='weight_kg', index = 'color')
# Pivot tables default to mean
# to change that use the aggfunc method
dogs.pivot_table(values='weight_kg', index = 'color', aggfunc=np.median)
# you can pass a list of functions too
dogs.pivot_table(values='weight_kg', index = 'color', aggfunc= [np.mean, np.median])

# pivot  on two variables
dogs.groupby(['color', 'breed'])['weight_kg'].mean()
dogs.pivot_table(values='weight_kg', index='color', columns='breed')

#instead of missing values you can have them filled in with a different value
dogs.pivot_table(values='weight_kg', index='color', columns='breed', fill_value=0)

#including margins = True will give the mean value at the end of each row
# the last row is the sum of each column 
dogs.pivot_table(values='weight_kg', index='color', columns='breed', fill_value=0, margins=True)


"""Explicit Indexes"""
# A dataframe has 3 types of data
# an array of data, an index of rows, and a set of columns

dogs.columns #returs a list of the column names in the data frame
# ->Index(['name', 'breed'..., 'weight_kg'], dtype = 'object')

dogs.index # returns a list of row numbers
# -> RangeIndex(start=0,stop=7,step=1)

#to move a column to become a row, you use the set_index method.
# This will now have the dogs names as the index
dogs_ind = dogs.set_index=('name')

#to undo changing a column use reset_index
dogs_ind.reset_index()

#to reset the index to numeric values AND get rid of the column you just created use:
dogs_ind.reset_index(drop=True)

dogs[dogs['name'].isin(['Bella', 'Stella'])]
# OR when the names at the index
dogs_ind.loc[['Bella', 'Stella']]

.loc uses idexes to find.
# The values don't need to be unique

#multi-level indexes sks hierarchical indexes
# color will be nested inside of breed (Breed is outter, color is inner)
dogs_ind3 = dogs.set_index(['breed', 'color'])

# For an outter level query, pass a list
dogs_ind3.loc[['Labrador', 'Chihuahua']]

# To subset on inner levels, you need to pass a list of touples
dogs_ind3.loc[[('Labrador', 'Brown'), ('Chihuahua', 'Tan')]]
# each row returned must match both parts of the touple to be returned

#You can sort by index values
dogs_ind3.sort_index()
#by default it sorts outter to inner in ascending order.
# You can control the sorting by passing in lists to the level and ascending arguments
dogs_ind3.sort_index(level=['color', 'breed'], ascending=[True, False])


"Slicing lists"
breeds[2:5]
breeds[:3]
breeds[:] #whole list

dogs_srt = dogs.set_index(['breed', 'color']).sort_index()
#this will sort by breed, the outter level

dogs_srt.loc['Chow Chow': "Poodle"] #will grab the rows that include Chow Chow to Poodle.

#You cannot use .loc[] to slice on inner idex levels with a list.

#You should use touples to slice on inner indexes
dogs_srt.loc[('Labrador', 'Brown'):('Schnauzer', 'Grey')]

#Slicing Columns requires two arguments
dogs_srt.loc[: , 'name':'height_cm']

#Slicing on both rows AND columns
dogs_srt.loc[('Labrador','Brown'):('Schanauzer','Grey') , 'name':'height_cm']

# First we'll index by date of birth and sort it.
dogs = dogs.set_index('date_of_birth').sort_index()

dogs.loc['2014-08-25':'2016-09-16']
#They also take partial dates
dogs.loc['2014':'2016']

#.iloc uses row/column index numbers
print(dogs.iloc[2:5, 1:4])



"""subsetting on piivot tables"""
dogs_height_by_breed_vs_color = dog_pack.pivot_table("height_cm", index = 'breed', columns = 'color')
# height_cm is the data we'll be looking at
# It'll be grouped in rows of 'breed'
# It'll have columns of 'color'

# .loc + slicing
dogs_height_by_breed_vs_color.loc["Chow Chow": "Poodle"]

#summary statistics default to the index.
# which is by row
dogs_height_by_breed_vs_color.mean(axis='index')

# to get a summary statistic accross columns use the 'columns' argument
dogs_height_by_breed_vs_color.mean(axis='columns')


"""visualizing data"""
import matplotlib.pyplot as plt

# histogram
dog_pack['height_cm'].hist()
plt.show()

#use the bins argument to adjust the number of buckets for the histogram
dog_pack['height_cm'].hist(bins=20)
plt.show()

avg_weight_by_breed = dog_pack.groupby('breed')['weight_kg'].mean()
print(avg_weight_by_breed)

#bar plot
avg_weight_by_breed.plot(kind='bar', title = 'Mean Weight by Dog Breed')
plt.show()

# line plots are good for numerical values over time.
sully.head()
sully.plot(x='date', y='weight_kg', kind='line')
plt.show()

# rotate the laels on the x-axis by passing in the rot argument with a number of degrees.
sully.plot(x='date', y='weight_kg', kind='line', rot=45)


# Scatter plots are good for relationships between two numeric variables
dog_pack.plot(x='height_cm', y='weight_kg', kind='scatter')
plt.show()


#layering plots multiple graphs
dog_pack[dog_pack['sex'] == 'F' ]['height_cm'].hist(alpha=0.7)
dog_pack[dog_pack['sex'] == 'M ']['height_cm'].hist(alpha=0.7)
# use plt.legend to display which graph belongs to which color
plt.legend(['F', 'M'])
# adding an alpha to the end of the histogram will make the plot transparent
# 0 means completley invisible and 1 means completley solid
plt.show()


"""Missing Values """

# isna() is used to check for missing values
#returns a boolean value if a record is missing
dogs.isna()

# Gives you a True or False for each column stating "There is missing Data"
dogs.isna().any()

dogs.isna.sum() #gives you the number of missing values in each column

# Plotting missing values
import matlotlib.pyplot as plt
dogs.isna().sum().plot(kind='bar')
plt.show()

# We could drop rows with missing values
dogs.dropna()

#We could replace missing values. 
dogs.fillna(0) #will find all the missing values and replace them with 0.


"""creating data frames"""
# There are two ways to make a DataFrame
# 1. From a list of dictionaries, this is done row by row
# 2. From a dictionary of lists, this is done coumn by column

#1st method. list of dictionaries.
list_of_dictionaries = [
    {'name': 'Ginger', "breed": "Dachsund", "height_cm": 22,
    "weight_kg":10, "date_of_birth":"2019-03-14"},
    {'name': 'Scout', 'breed': 'Dalmation', 'height_cm': 59,
    "weight_kg":25, 'date_of_birth':'2019-05-09'}
]

new_dogs = pd.DataFrame(list_of_dicts)
print(new_dogs)


#2 dictionary of lists
dict_of_lists = {
    "name": ['Ginger', 'Scout'],
    'breed': ['Daschshund', 'Dalmation'],
    'height_cm' : [22, 59],
    'weight_kg' : [10, 25],
    'date_of_birth' : ['2019-03-14', '2019-05-09']
}

new_dogs = pd.DataFrame(dict_of_lists)


# Reading and writing to CSVs files
# CSV = comma-separated values
# designed for DataFrame-like data
# most database and spreadsheet programs can use them or create them

new_dogs.csv

import pandas as pd
new_dogs = pd.read_csv('new_dogs.csv')
print(new_dogs)

new_dogs['bmi'] = new_dogs['weight_kg']/ (new_dogs['height_cm']/100)** 2
print(new_dogs)


#create a csv file
new_dogs.to_csv('new_dogs_with_bmi.csv')




# Joining data with Pandas
# Merging data from different tables
"""
Data Merging Basics
.merge()
multiple tables
types of joins: left, right, indexes
filtering joins
combining tables with concat
.merge_ordered()
.merge_asof()
.melt()

Merging Tables With Different Join Types
Advanced Merging and Concatenating
Merging Ordered and Time-Series Data
"""

"""Joining"""
wards = pd.read_csv("Ward_Offices.csv")
print(wards.head())
print(wards.shape)
# Columns: ward, alderman, address, zip
# 1 row for each ward.

census = pd.read_csv('Ward_Census.csv')
print(census.head())
print(census.shape)
#Columns: ward, pop_2000, pop_2010, change, address, zip
# 50 rows, and 6 columns. 

# We can match them on the ward variable
wards_census = wards.merge(census, on='ward')
print(wards_census.head(4))
#since wards went first it's columns will be first on the output.

# A merge has only the columns where they both have data.
# 
print(wards_census.columns)
# some columns have a suffix like address_x, zip_x, address_y
# this is because both tables had them
wards_census = wards.merge(census, on'ward', suffixes=('_ward', '_cen'))
print(wards_census.head())
print(wards_census.shape)

"""Relationships in tables."""
# One to one: Each entry in the left table maps to eaxctly one value in the right table.
# One to many: Each entry in the left table may map to multipe entries on the right table.
liscenses = pd.read_csv('Business_Licenses.csv')

print(liscenses.head())
print(liscenses.shape)

# Each business has it's ward listed. When merging the two tables, 
#each ward on the wards table will coorespond to one entry per business liscence
ward_licenses = wards.merge(licenses, on='ward', suffixes=('_ward', '_lic'))
print(ward_licenses.head())


"""Merging Multiple DataFrames"""
# To merge on two or more values, pass in a list.
grants.merge(licenses, on=['address', 'zip'])

#To merge multiple tables at the same time
grants_licenses_ward = grants \
    .merge(licenses, on=['address','zip']).merge(wards, on='ward', suffixes =('_bus', '_ward'))

grants_licenses_ward.head()

import matplotlib.pyplot as plt
grant_licenses_ward.groupby('ward').agg('sum').plot(kind='bar', y='grant')
plt.show()

# for the tables
df1.merge(df2, on='col') \
    .merge(df3, on='col')

#for 4 tables etc
df1.merge(df2, on='col') \
    .merge(df3, on='col') \
    .merge(df4, on='col')

"""Left Join"""
# returns all rows of data from the left, and where the values in the right match.
# how Defaults to inner
movies_taglines = movies.marge(taglines, on ='id', how='left')
print(movies_taglines.head()) 


# Merge the movies table with the financials table with a left join
movies_financials = movies.merge(financials, on='id', how='left')
# Count the number of rows in the budget column that are missing
number_of_missing_fin = movies_financials['budget'].isnull().sum()
# Print the number of movies missing financials
print(number_of_missing_fin)


#If you're joining tables on columns that are named differently, you can use
left_on = '' and right_on =''

tv_movies = movies.merge(tv_genre, how='right', left_on='id', right_on='movie_id')

#outter join returns all of the rows regarless if they have matches or not.

# Use right join to merge the movie_to_genres and pop_movies tables
genres_movies = movie_to_genres.merge(pop_movies, how='right', left_on = 'movie_id', right_on= 'id')

# Count the number of genres
genre_count = genres_movies.groupby('genre').agg({'id':'count'})

# Plot a bar chart of the genre_count
genre_count.plot(kind='bar')
plt.show()


"""Merging a table to itself
Common Situations:
    * hierarchical relatipship
    * sequential relationships
    * graph data
"""
#This will only show movies that have a sequal
original_sequals = sequals.merge(sequals, left_on='sequal', right_on='id', \
    suffixes = ('_org', '_seq'))
print(original_sequals.head())

#Using a left join, we'll see all of the movies AND their sequals
original_sequals = sequals.merge(sequals, how='left', left_on='sequal', right_on='id', \
    suffixes = ('_org', '_seq'))


"""Merging tables on idexes"""
# This makes a data frame so that the index is the ID
movies = pd.read_csv('imdb_movies.csv', index_col=['id'])


movies_genres = movies.merge(movie_to_genres, left_on='id', left_index = True \
    right_on='movie_id', right_index = True)


# Merge sequels and financials on index id
sequels_fin = sequels.merge(financials, on='id', how='left')

# Self merge with suffixes as inner join with left on sequel and right on id
orig_seq = sequels_fin.merge(sequels_fin, how='inner', left_on='sequel', 
                             right_on='id', right_index=True,
                             suffixes=('_org','_seq'))

# Add calculation to subtract revenue_org from revenue_seq 
orig_seq['diff'] = orig_seq['revenue_seq'] - orig_seq['revenue_org']

# Select the title_org, title_seq, and diff 
titles_diff = orig_seq[['title_org','title_seq','diff']]

# Print the first rows of the sorted titles_diff
print(titles_diff.sort_values('diff', ascending = False).head())


"""filtering joins"""

#Semi-Join
# Semi-Join returns the intersection, similar to an inner join.
# Returns on columns from the left table and not the right.
# No Duplicates

genres['gid'].isin(genres_tracks['gid']) #returns a boolean table of true and false

# The semi join
genres_tracks = genres.merge(top_tracks, on='gid') #left join of genres and top tracks
top_genres = genres[genres['gid'].isin(genres_tracks['gid'])]
print(top_genres.head())


#Anti-Join
# Returns the left table, excluding the intersection
# Returns the entries from the left table that are not in the right.
# Returns only columns from the left table, and not the right.

genres_tracks = genres.merge(top_tracks, on='gid', how='left', indicator=True)
# the indicator=True column returns a _merge column ie: both, left_only
gid_list = genres_tracks.loc[genres_tracks['_merge'] == 'left_only', 'gid']
non_top_genres = genres[genres['gid'].isin(gid_list)]



"concatenating tables vertically"
concat()

# Suppose you have multiptiple tables that contain information
inv_jan, inv_feb, inv_mar
# They have the same 
pd.concat([inv_jan, inv_feb, inv_mar])
axis = 0 by default

# If the index doesn't have valuable information, set ignore_index=True
# To reset the index numbers ignore_index = True

#when using keys, set the ignore index argument to False 
pd.concat([inv_jan,inv_feb,inv_mar],
            ignore_index=False,
            keys-['jan','feb','mar'])

# Combining tables with different columns. (In this case Feb has an extra column)
pd.concat([inv_jan, inv_feb],
    sort=True)

#If we only want the matching columns from different tables, we set the join to 'inner'
pd.concat([inv_jan, inv_feb], join='inner')

.append()
# simplified concat
# supports ignore_index and sort
# does not support keys and join
    # ALWAYS join = outer

inv_jan.append([inv_feb, inv_mar],
                ignore_index = True,
                sort = True)



# Concatenate the tables and add keys
inv_jul_thr_sep = pd.concat([inv_jul, inv_aug, inv_sep], 
                            keys=['7Jul', '8Aug', '9Sep'])
print(inv_jul_thr_sep.head())
# Group the invoices by the index keys and find avg of the total column
avg_inv_by_month = inv_jul_thr_sep.groupby(level=0).agg({'total': 'mean'})

# Bar plot of avg_inv_by_month
avg_inv_by_month.plot(kind='bar')
plt.show()


"Verifying integrity of data"
.merge(validate=None)
'one_to_one'
'one_to_many'
'many_to_one'
'many_to_many'

tracks.merge(specs, on='tid', validate ='one_to_one')

#inside concat
verify_integrity=False #checks for unique index values

pd.concat([inv_feb,inv_mar], verify_integrity=True)


"""
"Merge Ordered"
# When to use
# Ordered data/time series
# Filling in Missing Data



.merge()                                VS          .merge_ordered()
* Coulums(s) to join on each table                  * Coulums(s) to join on each table
    on, left_on, right_on                               on, left_on, right_on                   
* Type of join                                      * Type of join
    how (left, right, inner, outer)                     how (left, right, inner, outer)
    default to inner                                    default to outer
* Overlapping column names                          * Overlapping column names
    suffixes = []                                           suffixes =[]
*calling the method                                 *calling the method
    df1.merge(df2)                                      pd.merge_ordered(df1,df2)
"""

# appl has stock date and close
#  mcd has stock date and close
import pandas as pd
pd.merge_ordered(appl, mcd, on='date', suffixes=('_appl', '_mcd'))

#forward fill fills in the rows with missing dates from the previous rows
pd.merge_ordered(appl, mcd, on='date', suffixes=('_appl', '_mcd'), fill_method='ffill')


print (gdp_returns.corr())
#Something is correllated if it's 


"""
merge_asof()
Use this when 
* data sampled from a process and dates and times don't exactly align
* time series  training set
* developing a training set (no data leakage)


similar to a merge_ordered() left join
Will match on the nearest key column and not exxact matches.

on columns MUST be sorted
"""
# It's a left join so we see all the values from the visa table.
# The filling of values defaults to "Less than or equal"
pd.merge_asof(visa,ibm, on='date_time', suffixes=('_visa','_ibm'))


# using forward, it grabs the next value that is greater than or equal.
pd.merge_asof(visa,ibm, on='date_time', suffixes=('_visa','_ibm'), direction='forward')

# Using direction = 'nearest' will take the values that is closest to it regarless of direction.




# Eaxmple graphing some stuff.
# Merge gdp and recession on date using merge_asof()
gdp_recession = pd.merge_asof(gdp, recession, on='date')
print(gdp_recession.head(8))
# Create a list based on the row value of gdp_recession['econ_status']
is_recession = ['r' if s=='recession' else 'g' for s in gdp_recession['econ_status']]
print(is_recession)
# Plot a bar chart of gdp_recession
gdp_recession.plot(kind='bar', y='gdp', x='date', color=is_recession, rot=90)
plt.show()


# method to select data
# Similar to the so the statement after the WHERE clause in the SQL statement
.query()

# Example
stocks.query('nike>=90')
stocks.query('nike>=90 and disney < 90')
stocks.query('nike>=90 or disney < 90')

stocks_long.query('stock=="disney" or (stock=="nike" and close < 90)')
# When checking text, you must use double =, ie == 
# We also use double quotes to avoid confusion with the interpreter.

"""melt method"""
.melt() # unpivots a table

social_fin_tall = social_fin.melt(id_vars = ['financial','company']) 
#The ID_VARS are the columns in the data set you DO NOT WANT TO CHANGE
print(social_fin_tall.head(10))

social_fin_tall = social_fin.melt(id_vars=['financial','company'], value_vars=['2018', '2017'])
# note that the order of value_vars is preserved. 

"""
Don't even trip dawg. This melt method will take a table and make it tall. 
That means that it will keep whatever columns you want and put all of the other ones into
variable value relationships.
If you want to keep column A, and D after the melt
B and C will be in the Variable column and their values will coorespond to the value column

"""

social_fin_tall = social_fin.melt(id_vars=['financial','company'], value_vars=['2018', '2017'] \
                    var_name=['year'], value_name = 'dollars')




value_counts()
sort_index()
describe() # gives summary statistics
replace(['values you want to replace'], 'replacing it with') #np.nan is what we're replacing it with.
ex:pounds = pounds.replace([98,99], np.nan) # this finds the values 98 and 99 and replaces them with NaN

#You can call replace() with inplace=True and you won't need to save the data frame as a new variable.
ex: ounces.replace([98,99], np.nan, inplace = True)





# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace(8, np.nan, inplace = True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())


# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())


# Histogram
plt.hist(birth_weight.dropna(), bins=30) #dropna() drops all of the nan values
plt.xlabel('Birth weight (lb)')
plt.ylabel('Fraction of births')
plt.show()

preterm = nsfg['prglngth'] < 37
preterm.sum() #returns a boolean of True and False or 1 and 0
preterm.mean() #gives the average number of True values

preterm_weight = birth_weight[preterm]
full_term_weight = birth_weight[~preterm]
& is and
| is or

resample_rows_weighted()


plt.hist(agecon, bins=20, histtype='step')
# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')
# Show the figure
plt.show()



# Create a Boolean Series for full-term babies
full_term = nsfg['prglngth'] >= 37
# Select the weights of full-term babies
full_term_weight = birth_weight[full_term] 
# Compute the mean weight of full-term babies
print(full_term_weight.mean())


# Filter full-term babies
full_term = nsfg['prglngth'] >= 37
# Filter single births
single = nsfg['nbrnaliv'] == 1
# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[full_term & single]
print('Single full-term mean:', single_full_term_weight.mean())
# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[full_term & ~single]
print('Multiple full-term mean:', mult_full_term_weight.mean())



gss = pd.read_hdf('gss.hdf5', 'gss')
gss.head()

educ = gss['educ']
plot.hist(educ.dropna(), label = 'educ')
plt.show()

# PMF probability Mass function
pmf_educ = Pmf(educ, normalize=False)
pmf_head.head()
pmf_educ[12]

normalize = True # means that all of the values will be scaled down to equal 1. So they'll be
# given in percentages.

pmf_educ.bar(label='educ')
plt.xlabel('Years of education')
plt.ylabel('PMF')
plt.show()

# CDF Cumulative distribution functions

# PMF gives proability of x
# CDF gives proability of values less than or equal to x

cdf = Cdf(gss['age'])
cdf.plot()
plt.xlabel('Age')
plt.ylabel('CDF')
plt.show()

# evaluating a cdf
q = 51
p = cdf(q) # returns the cooresponding probability.
print(p)

#the inverse also works.
p = .25
q = cdf.inverse(p)
print(q)



# Select realinc
income = gss['realinc']
# Make the CDF
cdf_income = Cdf(income)
# Plot it
cdf_income.plot()
# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.show()


#Comparing distributions
#plot multiple distributions
male = gss['sex'] == 1
age= gss['age']
male_age = age[male]
female_age = [age[~male]]

#multiple Pmf plots 
Pmf(male_age).plot(label='Male')
Pmf(female_age).plot(label='Female')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()

# multiple Cdf plots
Cdf(male_age).plot(label='Male')
Cdf(female_age).plot(label='Female')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()


# Select educ
educ = gss['educ']
# Bachelor's degree
bach = (educ >= 16)
# Associate degree
assc = (educ >= 14) & (educ < 16)
# High school (12 or fewer years of education)
high = (educ <= 12)
print(high.mean())


# Plotting
income = gss['realinc']
# Plot the CDFs
Cdf(income[high]).plot(label='High school')
Cdf(income[assc]).plot(label='Associate')
Cdf(income[bach]).plot(label='Bachelor')
# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.legend()
plt.show()


# Modeling Probability Distributions 
# The normal distribution

sample = np.random.normal(size=1000)
Cdc(sample.plot())

from scipy.stats import norm
# CDF Example
xs = np.linspace(-3,3)
ys = norm(0,1).cdf(xs)
plt.plot(xs, ys, color='gray')
Cdf(sample).plot()

#PDF example
xs = np.linspace(-3,3)
ys = norm(0,1).pdf(xs)
plt.plot(xs, ys, color='gray')

# Kernel Density Estimation
# You can use the points in a sample to estimate the PDF of the distribution they came from. 
# This process is called Kernel Density Estimation or KDE
# It's how you go from a probability mass function to a probability density function
import seaborn as sns
sns.kdeplot(sample) #estimates the PDF and plots it.

xs = np.linspace(-3,3)
ys = norm(0,1).pdf(xs)
plt.plot(xs, ys, color='gray')
sns.kdeplot(sample)

"""
Use CDFs for exploration, however they're less well known
use PMFs if there are a small number of unique values
Use KDE if there are a lot of values
"""


# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income
Cdf(log_income).plot()

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()



# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income)

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show() 



# PCA() is used for priciple component analysis

state_pca = PCA().fit(state_summary_scale)






"""
Importing Data from a ton of different files.
# plain text files
# Reading a flat file or csv using NumPy or pandas
# importing excel spreadsheets
# pickled files
# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA
# Importing HDF5 files
# Hierarchical Data Format 5
# MATLAB and .mat files
# SQL Queries in Python
# Webdata from a URL
# webscraping using requests and urllib
# scraping using beautiful soup
# APIs (Application Programming Interface)
# JSONs (JavaScript Object Notation)
# tweepy 
# html parsing
"""

Importing data from a variety of sources

plain text files

table data

# reading a text or txt file
filename = 'fuck_finn.txt'
file = open(filename, mode='r') #r is to read(
text = file.read()
file.close()

print(file)
with open('huck_finn.txt', 'r') as file:  #context manager
    print(file.read())


# Open a file: file
file = open('moby_dick.txt', 'r')
print(file.read())
# Print it
print(file)
# Check whether file is closed
print(file.closed)
# Close file
file.close()
# Check whether file is closed
print(file.closed)

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

# Reading a flat file or csv
# NumPy or pandas
loadtxt() #breaks down when there is numeric data
import numpy as np
filename = 'MNIST_Header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols[0,2], dtype=str) 
# delimiter defaults to white space
# skiprows is in case you have a header
# usecols is what columns of the daya you want to use.
# dtype=str
print(data)

# Import package
import numpy as np
# Assign filename to variable: file
file = 'digits.csv'
# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')
# Print datatype of digits
print(type(digits))
# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))
# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()


# Import numpy
import numpy as np
# Assign the filename: file
file = 'digits_header.txt'
# Load the data: data
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2])
# '\t' is tab delimited
# skip the first row
# we grabbed the first and 3rd columns


# Assign filename: file
file = 'seaslug.txt'
# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)
# Print the first element of data
print(data[0])
# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)
# Print the 10th element of data_float
print(data_float[9])
# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()


data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
# first is filename
# second is comma delimeter
# names asks if there is a header
# dtype asks if we want to specify a data type or let the function figure it out.

#read from a csv
np.recfromcsv() # behaves similarly to read from csv and read from text.
# defaults to delimiter = ','
# defalts to names = True
# defaults to dtype = None

#importing flat files using pandas
import pandas as pd
filename = 'winequality-red.csv'
data = pd.read_csv(filename)
data.head()


# Assign the filename: file
file = 'digits.csv'
# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header= None)
# Build a numpy array from the DataFrame: data_array
data_array = np.array(data)
# Print the datatype of data_array to the shell
print(type(data_array))

# importing a bad file from pandas
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Assign filename: file
file = 'titanic_corrupt.txt'
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values=['NA', 'NaN', 'Nothing'])
# sep #separator
# comment is the character for commenting
# na_values is a list of strings that are null

# Print the head of the DataFrame
print(data.head())
# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()

# pickled files
# many data types
import pickle
with open('pickled_fruit.pkl', 'rb') as file: 
#rb is read only in binary
    data = pickle.load(file)
print(data)

#importing excel spreadsheets
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)

df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float you want to load as a data frame



# Import pandas
import pandas as pd
# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xls
xls = pd.ExcelFile(file)
# Print sheet names
print(xls.sheet_names)
# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')
# Print the head of the DataFrame df1
print(df1.head())
# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)
# Print the head of the DataFrame df2
print(df2.head())




# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA

# common SAS files are Cat and Dat which are catalogue and Data Set files
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sass7bdat') as file:
    df_sas = file.to_data_frame()

# Example
# Import sas7bdat package
from sas7bdat import SAS7BDAT
# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
# Print head of DataFrame
print(df_sas.head())
# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()



#importing stata files
import pandas as pd
data = pd.read_stata('urbanpop.dta')

# Import pandas
import pandas as pd
# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')
# Print the head of the DataFrame df
print(df.head())
# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()


# Importing HDF5 files
# Hierarchical Data Format 5
# Standard for storing large quantities of numerical data

import h5py
filename = 'whatever'
data = h5py.File(filename, 'r') #r is to read
print(type(data))

for key in data.keys():
    print(key)

"""meta- meta data for the file
quality- referes to data quality.
strain- strain data from the interferometer. In some sense, this is "the data"""

for key in data['meta'].keys():
    print(key)

print(np.array(data['meta']['Description']), np.array(data['meta']['Detector']))


# Import packages
import numpy as np
import h5py
# Assign filename: file
file = 'LIGO_data.hdf5'
# Load file: data
data = h5py.File(file, 'r')
# Print the datatype of the loaded file
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)


# Get the HDF5 group: group
group = data['strain']
# Check out keys of group
for key in group.keys():
    print(key)
# Set variable equal to time series data: strain
strain = np.array(data['strain']['Strain'])
# Set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()


# MATLAB and .mat files
scipy.io.loadmat() - read .mat files
scipy.io.savemat() - write .mat files

import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))

"""keys = matlab variable names
values = objects asssigned to variables"""



# Import package
import scipy.io
# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')
# Print the datatype type of mat
print(type(mat))
# Print the keys of the MATLAB dictionary
print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))
# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()


# Relational data bases
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Nothwind.sqlite')
table_names = engine.table_names()
print(table_names)


# Import necessary module
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Save the table names to a list: table_names
table_names = engine.table_names()
# Print the table names to the shell
print(table_names)


"""
WORKFLOW OF SQL QUERYING
- import packages and functions
- create the database engine
- connect to the engine
- Query the database
- Save query results to a DataFrame
- Close the connection
"""

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Orders")
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

print(df.head())

# You can use the context manager
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect as con:
    rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size=5))
    df.columns = rs.keys()


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()
# Print the length of the DataFrame df
print(len(df))
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee where EmployeeId >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())
    # Set the DataFrame's column names
    df.columns = rs.keys()
# Print head of DataFrame
print(df.head())



#pandas can do this in one line.
df = pd.read_sql_query("SELECT * FROM Orders", engine)



# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Album", engine)
# Print head of DataFrame
print(df.head())
# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
# Confirm that both methods yield the same result
print(df.equals(df1))


# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Employee WHERE EmployeeId >= 6 Order by Birthdate",engine)
# Print head of DataFrame
print(df.head())


# INNER JOIN in Pandas
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
df = pd.read_sql_query("SELECT OrderID, CompanyName From Orders \
    INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID", engine)
print(df.head())


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print head of DataFrame df
print(df.head())


# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000",engine)
# Print head of DataFrame
print(df.head())


# urllib
urlopen() # Accepts URLs instead of file names

from urllib.request import urlretrieve
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve(url, 'winequality-white.csv') # write this url to a file


#Reading csvs from the web or url using urlretrieve

# Import package
from urllib.request import urlretrieve
# Import pandas
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
urlretrieve(url, 'winequality-red.csv')
# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())


#Using pandas and a url to a csv from the web and save it as a data frame without saving it locally.

# Import packages
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')
# Print the head of the DataFrame
print(df.head())
# Plot first column of df
df.iloc[:, 0].hist()
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


# Importing non-flat files from the web

# Importing Excel files from the web 
# Import package
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/course/importing_data_into_r/latitude.xls'
# Read in all sheets of Excel file: xls
xls = pd.read_excel(url, sheet_name = None)
# Print the sheetnames to the shell
print(xls.keys())
# Print the head of the first sheet (using its name, NOT its index)
print(xls['1700'].head())


# HTTP Requests to import files from the web
from urllib.request import urlopen, Request
url = "http://www.wikipedia.org/"
request = Request(url)
response = urlopen(request) 
html = response.read()
response.close()

# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request: request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Print the datatype of response
print(type(response))
# Be polite and close the response!
response.close()


# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Extract the response: html
html = response.read()
# Print the html
print(html)
# Be polite and close the response!
response.close()




import requests
url = "http://www.wikipedia.org/"
r = requests.get(url)
text = r.text


# Import package
import requests
# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"
# Packages the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response: text
text = r.text
# Print the html
print(text)


# Beautiful Soup
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BeautifulSoup'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
print(soup.prettify())

soup.title
soup.get_text()
soup.find_all()

for link in soup.find_all('a'):
    print(link.get('href'))


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()
# Print the response
print(pretty_soup)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Get the title of Guido's webpage: guido_title
guido_title = soup.title
# Print the title of Guido's webpage to the shell
print(guido_title)
# Get Guido's text: guido_text
guido_text = soup.get_text()
# Print Guido's text to the shell
print(guido_text)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Print the title of Guido's webpage
print(soup.title)
# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


# APIs and JSONs
# JSON are dictionaries
import json
with open('snakes.json', 'r') as json_file:
    json_data = json.load(json_file)
type(json_data) # dictionary
for key, value in json_data.items():
    print(key + ':', value)


# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


import requests
url = 'http://www.omdbapi.com/?t=hackers'
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
    print(key + ':', value)



# Import requests package
import requests
# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Print the text of the response
print(r.text)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# Import package
import requests
# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


# Twitter
tweets.py
import tweepy, json
access_token = "..."
access_token_secret = '...'
consumer_key = '...'
consumer_secret = '...'

stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
stream.filter(track=['apples', 'oranges'])


# Store credentials in relevant variables
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
# Create your Stream object with credentials
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
# Filter your Stream variable
stream.filter(['clinton', 'trump', 'sanders', 'cruz'])


# Twitter API
# Import package
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Print the keys of the first tweet dict
print(tweets_data[0].keys())
# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
# Print head of DataFrame
print(df.head())

import re
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False


# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot the bar chart
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()











"""
Importing Data from a ton of different files.
# plain text files
# Reading a flat file or csv using NumPy or pandas
# importing excel spreadsheets
# pickled files
# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA
# Importing HDF5 files
# Hierarchical Data Format 5
# MATLAB and .mat files
# SQL Queries in Python
# Webdata from a URL
# webscraping using requests and urllib
# scraping using beautiful soup
# APIs (Application Programming Interface)
# JSONs (JavaScript Object Notation)
# tweepy 
# html parsing
"""

Importing data from a variety of sources

plain text files

table data

# reading a text or txt file
filename = 'fuck_finn.txt'
file = open(filename, mode='r') #r is to read(
text = file.read()
file.close()

print(file)
with open('huck_finn.txt', 'r') as file:  #context manager
    print(file.read())


# Open a file: file
file = open('moby_dick.txt', 'r')
print(file.read())
# Print it
print(file)
# Check whether file is closed
print(file.closed)
# Close file
file.close()
# Check whether file is closed
print(file.closed)

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

# Reading a flat file or csv
# NumPy or pandas
loadtxt() #breaks down when there is numeric data
import numpy as np
filename = 'MNIST_Header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols[0,2], dtype=str) 
# delimiter defaults to white space
# skiprows is in case you have a header
# usecols is what columns of the daya you want to use.
# dtype=str
print(data)

# Import package
import numpy as np
# Assign filename to variable: file
file = 'digits.csv'
# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')
# Print datatype of digits
print(type(digits))
# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))
# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()


# Import numpy
import numpy as np
# Assign the filename: file
file = 'digits_header.txt'
# Load the data: data
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2])
# '\t' is tab delimited
# skip the first row
# we grabbed the first and 3rd columns


# Assign filename: file
file = 'seaslug.txt'
# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)
# Print the first element of data
print(data[0])
# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)
# Print the 10th element of data_float
print(data_float[9])
# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()


data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
# first is filename
# second is comma delimeter
# names asks if there is a header
# dtype asks if we want to specify a data type or let the function figure it out.

#read from a csv
np.recfromcsv() # behaves similarly to read from csv and read from text.
# defaults to delimiter = ','
# defalts to names = True
# defaults to dtype = None

#importing flat files using pandas
import pandas as pd
filename = 'winequality-red.csv'
data = pd.read_csv(filename)
data.head()


# Assign the filename: file
file = 'digits.csv'
# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header= None)
# Build a numpy array from the DataFrame: data_array
data_array = np.array(data)
# Print the datatype of data_array to the shell
print(type(data_array))

# importing a bad file from pandas
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Assign filename: file
file = 'titanic_corrupt.txt'
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values=['NA', 'NaN', 'Nothing'])
# sep #separator
# comment is the character for commenting
# na_values is a list of strings that are null

# Print the head of the DataFrame
print(data.head())
# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()

# pickled files
# many data types
import pickle
with open('pickled_fruit.pkl', 'rb') as file: 
#rb is read only in binary
    data = pickle.load(file)
print(data)

#importing excel spreadsheets
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)

df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float you want to load as a data frame



# Import pandas
import pandas as pd
# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xls
xls = pd.ExcelFile(file)
# Print sheet names
print(xls.sheet_names)
# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')
# Print the head of the DataFrame df1
print(df1.head())
# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)
# Print the head of the DataFrame df2
print(df2.head())




# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA

# common SAS files are Cat and Dat which are catalogue and Data Set files
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sass7bdat') as file:
    df_sas = file.to_data_frame()

# Example
# Import sas7bdat package
from sas7bdat import SAS7BDAT
# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
# Print head of DataFrame
print(df_sas.head())
# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()



#importing stata files
import pandas as pd
data = pd.read_stata('urbanpop.dta')

# Import pandas
import pandas as pd
# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')
# Print the head of the DataFrame df
print(df.head())
# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()


# Importing HDF5 files
# Hierarchical Data Format 5
# Standard for storing large quantities of numerical data

import h5py
filename = 'whatever'
data = h5py.File(filename, 'r') #r is to read
print(type(data))

for key in data.keys():
    print(key)

"""meta- meta data for the file
quality- referes to data quality.
strain- strain data from the interferometer. In some sense, this is "the data"""

for key in data['meta'].keys():
    print(key)

print(np.array(data['meta']['Description']), np.array(data['meta']['Detector']))


# Import packages
import numpy as np
import h5py
# Assign filename: file
file = 'LIGO_data.hdf5'
# Load file: data
data = h5py.File(file, 'r')
# Print the datatype of the loaded file
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)


# Get the HDF5 group: group
group = data['strain']
# Check out keys of group
for key in group.keys():
    print(key)
# Set variable equal to time series data: strain
strain = np.array(data['strain']['Strain'])
# Set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()


# MATLAB and .mat files
scipy.io.loadmat() - read .mat files
scipy.io.savemat() - write .mat files

import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))

"""keys = matlab variable names
values = objects asssigned to variables"""



# Import package
import scipy.io
# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')
# Print the datatype type of mat
print(type(mat))
# Print the keys of the MATLAB dictionary
print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))
# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()


# Relational data bases
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Nothwind.sqlite')
table_names = engine.table_names()
print(table_names)


# Import necessary module
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Save the table names to a list: table_names
table_names = engine.table_names()
# Print the table names to the shell
print(table_names)


"""
WORKFLOW OF SQL QUERYING
- import packages and functions
- create the database engine
- connect to the engine
- Query the database
- Save query results to a DataFrame
- Close the connection
"""

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Orders")
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

print(df.head())

# You can use the context manager
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect as con:
    rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size=5))
    df.columns = rs.keys()


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()
# Print the length of the DataFrame df
print(len(df))
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee where EmployeeId >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())
    # Set the DataFrame's column names
    df.columns = rs.keys()
# Print head of DataFrame
print(df.head())



#pandas can do this in one line.
df = pd.read_sql_query("SELECT * FROM Orders", engine)



# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Album", engine)
# Print head of DataFrame
print(df.head())
# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
# Confirm that both methods yield the same result
print(df.equals(df1))


# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Employee WHERE EmployeeId >= 6 Order by Birthdate",engine)
# Print head of DataFrame
print(df.head())


# INNER JOIN in Pandas
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
df = pd.read_sql_query("SELECT OrderID, CompanyName From Orders \
    INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID", engine)
print(df.head())


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print head of DataFrame df
print(df.head())


# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000",engine)
# Print head of DataFrame
print(df.head())


# urllib
urlopen() # Accepts URLs instead of file names

from urllib.request import urlretrieve
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve(url, 'winequality-white.csv') # write this url to a file


#Reading csvs from the web or url using urlretrieve

# Import package
from urllib.request import urlretrieve
# Import pandas
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
urlretrieve(url, 'winequality-red.csv')
# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())


#Using pandas and a url to a csv from the web and save it as a data frame without saving it locally.

# Import packages
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')
# Print the head of the DataFrame
print(df.head())
# Plot first column of df
df.iloc[:, 0].hist()
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


# Importing non-flat files from the web

# Importing Excel files from the web 
# Import package
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/course/importing_data_into_r/latitude.xls'
# Read in all sheets of Excel file: xls
xls = pd.read_excel(url, sheet_name = None)
# Print the sheetnames to the shell
print(xls.keys())
# Print the head of the first sheet (using its name, NOT its index)
print(xls['1700'].head())


# HTTP Requests to import files from the web
from urllib.request import urlopen, Request
url = "http://www.wikipedia.org/"
request = Request(url)
response = urlopen(request) 
html = response.read()
response.close()

# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request: request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Print the datatype of response
print(type(response))
# Be polite and close the response!
response.close()


# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Extract the response: html
html = response.read()
# Print the html
print(html)
# Be polite and close the response!
response.close()




import requests
url = "http://www.wikipedia.org/"
r = requests.get(url)
text = r.text


# Import package
import requests
# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"
# Packages the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response: text
text = r.text
# Print the html
print(text)


# Beautiful Soup
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BeautifulSoup'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
print(soup.prettify())

soup.title
soup.get_text()
soup.find_all()

for link in soup.find_all('a'):
    print(link.get('href'))


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()
# Print the response
print(pretty_soup)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Get the title of Guido's webpage: guido_title
guido_title = soup.title
# Print the title of Guido's webpage to the shell
print(guido_title)
# Get Guido's text: guido_text
guido_text = soup.get_text()
# Print Guido's text to the shell
print(guido_text)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Print the title of Guido's webpage
print(soup.title)
# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


# APIs and JSONs
# JSON are dictionaries
import json
with open('snakes.json', 'r') as json_file:
    json_data = json.load(json_file)
type(json_data) # dictionary
for key, value in json_data.items():
    print(key + ':', value)


# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


import requests
url = 'http://www.omdbapi.com/?t=hackers'
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
    print(key + ':', value)



# Import requests package
import requests
# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Print the text of the response
print(r.text)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# Import package
import requests
# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


# Twitter
tweets.py
import tweepy, json
access_token = "..."
access_token_secret = '...'
consumer_key = '...'
consumer_secret = '...'

stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
stream.filter(track=['apples', 'oranges'])


# Store credentials in relevant variables
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
# Create your Stream object with credentials
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
# Filter your Stream variable
stream.filter(['clinton', 'trump', 'sanders', 'cruz'])


# Twitter API
# Import package
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Print the keys of the first tweet dict
print(tweets_data[0].keys())
# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
# Print head of DataFrame
print(df.head())

import re
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False


# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot the bar chart
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()








# Cleaning Data in python
# Diagnose dirty data
# Side Effects of dirty data
# Cleaning data

# 1.1 Example
# Make sure your variables are the correct data types.

sales = pd.read_csv('sales.csv')
sales.head(2)
sales.dtypes

sales.info()

#since all of the revenue entries are a string, 
sales['Revenue'].sum() #will return a large string

#first remove the $
sales['Revenue'] = sales['Revenue'].str.strip('$') # gets rid of the dollar sign
sales['Revenue'] = sales['Revenue'].astype('int')

assert sales['Revenue'].dtypes == 'int'

df['marriage_status'].describe()
# convert to category
df['marriage_status'] = df['marriage_status'].astype('category')
df.describe()

# 1.1 Work
# Print the information of ride_sharing
print(ride_sharing.info())
# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())
# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')
# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'
# Print new summary statistics 
print(ride_sharing['user_type_cat'].describe())

# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip("minutes")
# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')
# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'
# Print formed columns and calculate average ride duration 
print(ride_sharing[['duration','duration_trim','duration_time']])
print(ride_sharing['duration_time'].mean())


 
# 1.2 Data in a range
# Example
movies.head()
import matplotlib.pyplot as plt
plt.hist(movies['avg_rating'])
plt.title('Average rating of movies (1-5)')

# Subscription dates in the future
import datetime as dt
today_date = dt.date.today()
user_signups[user_signups['subscription_date'] > today_date]

#dealing with out of range
"""
* drop it (shouldn't unless it's a small portion)
* Setting custom minimums and maximums
* treat as missing and impute
* setting custom value depending on business assumptions
"""
# isolate the movies with rating higher than 5.

# COULD Make a new data frame with movie values <= 5
movies = movies[movies['avg_rating'] <= 5]

# COULD drop the movies with a rating higher than 5.
movies.drop(movies[movies['avg_rating']> 5].index, inplace = True)
assert movies['avg_rating'].max() <= 5

# COULD convert ratings bigger than 5 to 5.
movies.loc[movies['avg_rating']> 5, 'avg_rating'] = 5

assert movies['avg_rating'].max() <= 5

import datetime as dt
import pands as pd
# output data types
user_signups.dtypes
#convert to date
user_signups['subscription_date'] = pd.to_datetime(user_signups['subscription_date']).dt.date
today_date = dt.date.today()
# Could either drop the data
# drop date using filtering
user_signups = user_signups[user_signups['subscription_date'] < today_date]
#OR
# drop values using .drop()
user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace = True)
# Hardcode dates with upper limit
# Drop values using filtering
user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
assert user_signups.subscription_date.max().date() <= today_date

# 1.2 WORK
# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')
# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27
# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')
# Print tire size description
print(ride_sharing['tire_sizes'].describe())
# Convert ride_date to date
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date']).dt.date
# Save today's date
today = dt.date.today()
# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today
# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())

#1.3 Duplicate values
# Example
# Find them
# height_weight.head()
duplicates = height_weight.duplicated()
print(duplicates) #boolian column
height_weight[duplicates]

#.duplicated()
# subset: list of column names to check for duplication
# keep: 'first', 'last', 'all' (to keep all use False) 

column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.duplicated(subset = column_names, keep = False)

height_weight[duplicates].sort_values(by='first_name')

.drop_duplicates()
# subset: list of column names
# keep: first, last, False (Defaluts to first.)
# inplace = True (drops them in the working dataframe)
height_weight.drop_duplicates(inplace = True)

# Output remaining duplicate values
column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.drop_duplicates(subset = column_names, keep = False)
height_weight[duplicates].sort_values(by= 'first_name')

# Treat duplicated values using the .groupby() and .agg() methods
# Group by column names and product statistical summaries
column_names = ['first_name', 'last_name', 'address']
# We're going to replace duplicated values in the height column with the maximum height.
# and replace duplicated weights with the average.
summaries = {'height': 'max', 'weight': 'mean'}
height_weight = height_weight.groupby(by = column_names).agg(summaries).reset_index()
# Make sure ggregation is done
duplicates = height_weight.duplicated(subset = column_names, keep = False)
height_weight[duplicates].sort_values(by = 'first_name')

# 1.3 Work
# Find duplicates
duplicates = ride_sharing.duplicated(subset = 'ride_id', keep = False)
# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')
# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id','duration','user_birth_year']])

# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()
# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}
# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby(by = 'ride_id').agg(statistics).reset_index()
# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]
# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0


# Text and Catagorical Data
# 2.1 Categories
# Drop Data
# remapping
# inferring categories
study_data = pd.read_csv('study.csv')
study_data #has blood types
categories #has values of the acceptable blood types

# First grab all inconsistent categories
inconsisten_categories = set(study_data['blood_type']).difference(catageories['blood_type'])
print(inconsistent_categories)
# returns categories in study data that are not in categories
inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
study_data[inconsistent_rows]
# Drop inconsistent categories and get consistent data only
consistent_data = study_data[~inconsistent_rows]

# 2.1
# Exercise
# Print categories DataFrame
print(categories)
# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")

# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])
# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)
# Print rows with inconsistent category
print(airlines[cat_clean_rows])
# Print rows with consistent categories only
print(airlines[~cat_clean_rows])

# 2.2 Categorical Variables
# Work

#Value consistency
# capitalization 'married', 'MARRIED', 'UNMARRIED', 'unmarried'

#get marriage status column
marriage_status = demographics['marriage_status']
marriage_status.value_counts()
# Get value counts on DataFrame
marriage_status.groupby('marriage_status').count()
# Solutions 
# Capitalize EX:
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
# Lowercase EX:
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
marriage_status['marriage_status'].value_counts()

# leading or trailing whitespace
marriage_status = demographics['marriage_status']
marriage_status.value_counts()
# strip() gets rid of all whitespace if left empty
demographics = demographics['marriage_status'].str.strip()
demographics['marriage_status'].value_counts()

#Collapsing data into categories

#Using qcut()
import pandas as pd
group_names = ['0-200K', '200K-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q=3, labels = group_names)
# print income_group column
demographics[['income_group', 'household_income']]
# this didn't work ^^^^

# using cut() - create category ranges and names
ranges = [0, 200000, 500000, np.inf]
group_names = ['0-200K', '200K-500K', '500K+']
# Create income group column
demographics['income_group'] = pd.cut(demographics['household_income'], bins = ranges, labels= group_names)
demographics[['income_group', 'household_income']]

# mapping multiple categories into fewer ones
Operating_system = ['Microsoft', 'MacOS', 'IOS', 'Android', 'Linux']
Operating_system = ['DesktopOS', 'MobileOS']

mapping = {'Microsoft': 'DesktopOS', 'MacOS': 'DesktopOS', 'Linux':'Desktop', 'IOS':'MobileOS', 'Android': 'MobileOS'}
devices['Operating_system'] = devices['Operating_system'].replace(mapping)
devices['operating_system'].unique()


#2.2 
# Exercies
# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())
# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower() 
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})
# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()
# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Create ranges for categories
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'medium', 'long']
# Create wait_type column
airlines['wait_type'] = pd.cut(airlines['wait_min'], bins = label_ranges, labels = label_names)
# Create mappings and replace
mappings = {'Monday':'weekday', 'Tuesday':'weekday', 'Wednesday': 'weekday', 
            'Thursday': 'weekday', 'Friday': 'weekday', 
            'Saturday': 'weekend', 'Sunday': 'weekend'}
airlines['day_week'] = airlines['day'].replace(mappings)



# 2.3  Cleaning Text Data
phone = pd.read_csv('phones.csv')
print(phone)

#Replace "+" with "00"
phone['Phone number'] = phone['Phone number'].str.replace("+", "00")
phone
# Replace "-" with nothing
phone['Phone number'] = phone['Phone number'].str.replace("-", "")
phone
#Changing all the phone numbers with fewer than 10 digits to NAN
digits = phone['Phone number'].str.len()
phone.loc[digits< 10, "Phone number"] = np.nan
phone

#Find the length of each row in Phone number column
sanity_check = phone['Phone number'].str.len()
# Assert minimum phone number length is 10
assert sanity_check.min() >= 10
# Assert all numbers do not have "+" or "-"
assert phone['Phone number'].str.contains('+'|'-').any() == False


# regular expression
# replace letters with nothing
phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')
phones.head()

# 2.3
# Exercise

# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.","")
# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Mr.","")
# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Miss","")
# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Ms.","")
# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False

# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()
# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]
# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40
# Print new survey_response column
print(airlines_survey['survey_response'])


# 3 Advanced Data Problems
# 3.1 Uniformity (same units on measurements)
# Examples

# Temperature Example
temperatures = pd.read_csv('temperature.csv')
temperatures.head()
"We see some in C and some in F"
" Let's look at a scatter plot"
import matplotlib.pyplot as plt
# Create a scatter plot
plt.scatter(x='Date', y='Temperature', data = temperatures)
plt.title('Temperature in Celsius March 2019 - NYC')
plt.xlabel('Dates')
ply.ylabel('Temperature in Celsius')
plt.show()

# C = (F-32) * (5/9)
# Convert all of the F temperatures to C
temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels
assert temperatures['Temperature'].max() < 40


# Birthday Example
birthdays.head()
"See birthdays in different formats"
27/27/19            ??
03-29-19            MM-DD-YY
March 3rd, 2019     Month D, YYYY

# datetime is useful for date times.
pandas.to_datetime()
%d-%m-%Y
% c
%m-%d-%Y

#convert to datetime.
birthdays['Birthday'] = pd.to_datetime(birthday['Birthday'])
# if multiple formats, it won't work

birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'],
                        # Attempt to infer format of each date
                        infer_datetime_format = True,
                        errrors = 'coerce')
# You could also use dt.strftime
birthdays['Birthday'] = birthdays['Birthday'].dt.strftime('%d-%m-%Y')
birthdays.head()

# Ambihuous date data
# 2019-03-08 August or March?

# 3.1 Work

# Find values of acct_cur that are equal to 'euro'
acct_eu = banking['acct_cur'] == 'euro'
# Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1
# Unify acct_cur column by changing 'euro' values to 'dollar'
banking.loc[acct_eu, 'acct_cur'] = 'dollar'
# Assert that only dollar currency remains
assert banking['acct_cur'].unique() == 'dollar'

# Print the header of account_opened
print(banking['account_opened'].head())
# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                           # Infer datetime format
                                           infer_datetime_format = True,
                                           # Return missing value for error
                                           errors = 'coerce') 

# Print the header of account_opend
print(banking['account_opened'].head())
# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                           # Infer datetime format
                                           infer_datetime_format = True,
                                           # Return missing value for error
                                           errors = 'coerce') 
# Get year of account opened
banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')
# Print acct_year
print(banking['acct_year'])


# 3.2 Cross field validation
# Examples
# Cross field validation the use of multiple fields in a dataset to sanity check data integrity

sum_classes = flights[['economy_class', 'business_class', 'first_class']].sum(axis=1)
passenger_equ = sum_classes == flights['total_passengers']
# filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]

# with birthdays and ages
users['Birthday'] = pd.to_datetime(users['Birthday'])
today = dt.date.today()
age_manual = today.year - users['Birthday'].dt.year
age_equ = age_manual == users['Age']
inconsistent_age = users[~age_equ]
inconsistent_age = users[age_equ]

# 3.2
# Work
# Store fund columns to sum against
fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']
# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis=1) == banking['inv_amount']
# Store consistent and inconsistent data 
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]
# Store consistent and inconsistent data
print("Number of inconsistent investments: ", inconsistent_inv.shape[0])


# Store today's date and find ages
today = dt.date.today()
ages_manual = today.year - banking['birth_date'].dt.year
# Find rows where age column == ages_manual
age_equ = banking['age'] == ages_manual
# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]
# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])


# 3.3 Completeness and missing data
# Examples
# NA, nan, 0, .

airquality = pd.read_csv('airquality.csv')
print(airquality)
airquality.isna()
airquality.isna().sum()

import missingno as msno
import matplotlib as pyplot
msno.matrix(airquality)
plt.show()

missing = airquality[airqulity['CO2'].isna()]
complete = airquality[~airquality['Co2'].isna()]

complete.describe()
missing.describe()

# first sort by temperature, then visualize
sorted_airquality = airquality.sort_values(by = 'Temperature')
msno.matrix(sorted_airquality)
plt.show()

# Drop missing values
airquality_dropped = airquality.dropna(subset = ['CO2'])
airquality_dropped.head()
# OR 
# Replace with statistical measures
co2_mean = airquality['CO2'].mean()
airquality_imputed.airquality.fillna({'CO2': co2_mean})
airquality_imputed.head()

# 3.3
# Exericises

# Print number of missing values in banking
print(banking.isna().sum())
# Visualize missingness matrix
msno.matrix(banking)
plt.show()
# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]
# Sort banking by age and visualize
banking_sorted = banking.sort_values(by = 'age')
msno.matrix(banking_sorted)
plt.show()

# Drop missing values of cust_id
banking_fullid = banking.dropna(subset = ['cust_id'])
# Compute estimated acct_amount
acct_imp = banking_fullid['inv_amount'] * 5
# Impute missing acct_amount with corresponding acct_imp
banking_imputed = banking_fullid.fillna({'acct_amount':acct_imp})
# Print number of missing values
print(banking_imputed.isna().sum())


# Chapter 4
# Comparing Strings

# 4.1 Record Linkage
# Comparing Strings
# minimum edit distance.
from thefuzz import fuzz

#Compare reeding vs reading
fuzz.WRatio('Reeding', 'Reading')
# gives a score from 1-100

from thefuzz import process

# Define string and array of possible matches
string = 'Houston Rockets vs Los Angeles Lakers'
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets', "Houson vs Los Angeles, 'Heat vs Bulls"])

process.extract(string, choices, limit = 2)
# Returns a list of touples.
# [('Rockets vs Lakers', 86, 0), ('Lakers vs Rockets', 86, 1)]

# .replace to collapse 'eur' into 'Europe'

print(survey['state'].unique())
categories = ['California', 'New York']

for state in categories['state']:
    matches = process.extract(state, survey['state'], limit = survey.shape[0])
    for potential_match in matches:
        # if high similarity score
        if potential_match[1] >= 80:
            survey.loc[survey['state'] == potential_match[0], 'state'] = state


#4.1 
# Work
# Import process from thefuzz
from thefuzz import process
# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['cuisine_type'].unique()
# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit = len(unique_types)))
# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit = len(unique_types)))
# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit = len(unique_types)))

# Inspect the unique values of the cuisine_type column
print(restaurants['cuisine_type'].unique())
# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit=len(restaurants['cuisine_type']))
# Inspect the first 5 matches
print(matches[0:5])
# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))
# Iterate through the list of matches to italian
for match in matches:
  # Check whether the similarity score is greater than or equal to 80
  if match[1] >=80:
    # Select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
    restaurants.loc[restaurants['cuisine_type'] == match[0], 'cuisine_type'] = 'italian'


# Iterate through categories
for cuisine in categories:  
  # Create a list of matches, comparing cuisine with the cuisine_type column
  matches = process.extract(cuisine, restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))
  # Iterate through the list of matches
  for match in matches:
     # Check whether the similarity score is greater than or equal to 80
    if match[1] >= 80:
      # If it is, select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
      restaurants.loc[restaurants['cuisine_type'] == match[0]] = cuisine
# Inspect the final result
print(restaurants['cuisine_type'].unique())


# 4.2
# Examples
# Record linkage

# generating pairs
import recordLinkage

# Create indexing object
indexer = recordlinkage.Index()

# Generate pairs blocked on state
indexer.block('state')
pairs = indexer.index(census_A, census_B)
print(pairs)

# create a compare object
compare_cl = recordlinkage.Compare()

# Find exact matches for pairs of date_of_birth and state
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label='state')

# Find similar matches for pairs of surname and address_1 using string similarity
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

# Find matches
potential_matches = compare_cl.compute(pairs, census_A, census_B)
print(potential_matches)

potential_matches[potential_matches.sum(axis=1) => 2]

# 4.2 
# Work
# Create an indexer and object and find possible pairs
indexer = recordlinkage.Index()
# Block pairing on cuisine_type
indexer.block('cuisine_type')
# Generate pairs
pairs = indexer.index(restaurants, restaurants_new)

# Create a comparison object
comp_cl = recordlinkage.Compare()
# Find exact matches on city, cuisine_types 
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('cuisine_type', 'cuisine_type', label = 'cuisine_type')
# Find similar matches of rest_name
comp_cl.string('rest_name', 'rest_name', label='name', threshold = 0.8) 
# Get potential matches and print
potential_matches = comp_cl.compute(pairs, restaurants, restaurants_new)
print(potential_matches)


#4.3 Linking DataFrames
# Examples
# We have already created pairs of indexes
# We have compared based on exact matches of a couple of fields.
# We compared based on kind of matching on a few others.
# We listed potential matches.


potential_matches = compare_cl.compute(full_pairs, census_A, census_B)
# is a multi-level array. The first index column is census_A, the
# second column is census_B

# First we're going to put matches together where the rows match up
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
print(matches)
# The output is rows that are most likely duplicates
# our next step is to filter out those duplicates

# We are picking the second column, which represents census_B

matches.index # gives us the indecies.

# Get indicies from census_B only
duplicate_rows = matches.index.get_level_values(1)
print(census_B_index)

# Finding duplicates in census_B
census_B_duplicates = census_B[census_B.index.isin(duplicate_rows)]

# Finding NON-duplicates
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]

full_census = census_A.append(census_B_new)


# 4.3
# Work

# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)
# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]
# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)












# Data Science Toolbox 2

# Iterable: list, dictionary, etc of things that can be iterated on.
# Interator: object that is keeping track of the next thing 

# 1.1 Iterables
# Examples
iter()
next()

word = 'Da'
it = iter(word)
next(it)

word = 'Data'
it = iter(word)
print(*it)

* # called the splat method.

for key, value in pythonistas.items():
    print(key, value)

# over file connections
file = open('file.txt')
it = iter(file)
print(next(it)) # prints the first line
print(next(it)) # prints the second line

# 1.1 
# Work

# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
# Print each list item in flash using a for loop
for name in flash:
    print(name)
# Create an iterator for flash: superhero
superhero = iter(flash)
# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

# Create an iterator for range(3): small_value
small_value = iter(range(3))
# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))
# Loop over range(3) and print the values
for i in range(3):
    print(i)
# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))
# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))


# Create a range object: values
values = range(10,21)
# Print the range object
print(values)
# Create a list of integers: values_list
values_list = list(values)
# Print values_list
print(values_list)
# Get the sum of values: values_sum
values_sum = sum(values)
# Print values_sum
print(values_sum)


# 1.2 new functions
# Examples
enumerate()
zip()

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e)) # enumerate object
e_list = list(e)
print(e_list) # returns touples of index and avenger name

for index, value in enumerate(avengers):
    print(index, value)

for index, value in enumerate(avengers, start = 10): # This means 
    #that the count will start at 10 instead of 1 
    print(index, value)


avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(type(z))
z_list = list(z) # returns a list of touples of [('hawkeye', 'barton'), ('iron man', stark) etc]


avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
for z1, z2 in zip(avengers, names):
    print(z1, z2)

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(*z)

# 1.2 Work
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']
# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))
# Print the list of tuples
print(mutant_list)
# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)
# Change the start index
for index2, value2 in enumerate(mutants, start = 1) :
    print(index2, value2)

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))
# Print the list of tuples
print(mutant_data)
# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)
# Print the zip object
print(mutant_zip)
# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)
# Print the tuples in z1 by unpacking with *
print(*z1)
# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)
# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1) 
# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

# 1.3 Load Large files into memory
# for more data that can be held in memory

import pandas as pd
results = []
for chunk in pd.read_csv('data.csv', chunksize=1000):
    result.append(sum(chunk['x']))
total = sum(result)
print(total)

import pandas as pd
total = 0
for chunk in pd.read_csv('data.csv', chunksize=1000):
    total += sum(chunk['x'])
print(total)

# 1.3
# Work
# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)


# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}
    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):  

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict
# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')
# Print result_counts
print(result_counts)


# 2.1 List Comprehensions
nums = [12,8,21,3,16]
new_nums = []
for num in nums:
    new_nums.append(num+1)
print(new_nums)

# OR

nums = [12,8,21,3,16]
new_nums = [num+1 for num in nums]

result = [num for num in range(11)]
print(result)
# [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# iterable
# iterator variable
# output expression

# newsted loops
pairs_1 = []
for num1 in range(0,2):
    for num2 in range(6,8):
        pairs.append((num1, num2))
print(pairs_1)
[(0,6), (0,7), (1,6), (1,7)]

pairs_2 = [(num1, num2), for num1 in range(0,2) for num2 in range(6,8)]
print(pairs_2)

# 2.1
# Work
# Create list comprehension: squares
squares = [i ** 2 for i in range(10)]
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]
# Print the matrix
for row in matrix:
    print(row)

#2.2 Conditionals in comprehensions
[num ** 2 for num in range(10) if num % 2 == 0]
[0,4,16,36,64]

[num ** 2 if num %2 == 0 else 0 for num in range(10)]
[0,0,4,0,16,0,36,0,64,0]

# dictionaries 
pos_neg = {num: -num for num in range(9)}

# 2.2 
# Work

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]
# Print the new list
print(new_fellowship)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create list comprehension: new_fellowship
new_fellowship = [mem if len(mem) >= 7 else "" for mem in fellowship]
# Print the new list
print(new_fellowship)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create dict comprehension: new_fellowship
new_fellowship = {member: len(member) for member in fellowship}
# Print the new dictionary
print(new_fellowship)

#2.3 Generators
[2 * num for num in range(10)]

# Use () insead and it's a geneator
(2 * num for num in range(10))

result = (num for num in range(6))
for num in result:
    print(num)

result_list = list(result)

print(next(result)) # 0
print(next(result)) # 1
print(next(result)) # 2
print(next(result)) # 3
print(next(result)) # 4

(num for num in 10 * 1000000)

even_nums = (num for num in range(10) if num % 2 == 0)

# generator functions

def num_sequence(n):
    """generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1

result = num_sequence(5)
print(type(result)) # generator

for item in result:
    print(item)
0
1

# 2.3
# Work
# Create generator object: result
result = (num for num in range(31))
# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))
# Print the rest of the values
for value in result:
    print(value)


# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
# Create a generator object: lengths
lengths = (len(person) for person in lannister)
# Iterate over and print the values in lengths
for value in lengths:
    print(value)


# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""
    # Yield the length of a string
    for person in input_list:
        yield len(person)
# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)


# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']
# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]
# Print the extracted times
print(tweet_clock_time)


# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']
# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']
# Print the extracted times
print(tweet_clock_time)


# 3.1 Case Study
# Work

# Zip lists: zipped_lists
zipped_lists = zip(feature_names, row_vals)
# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)
# Print the dictionary
print(rs_dict)

# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return(rs_dict)
# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)
# Print rs_fxn
print(rs_fxn)


# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])



# Import the pandas package
import pandas as pd
# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]
# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)
# Print the head of the DataFrame
print(df.head())



# 3.2 case study
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)


# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))


# Initialize an empty dictionary: counts_dict
counts_dict = {}
# Open a connection to the file
with open('world_dev_ind.csv') as file:
    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):
        row = line.split(',')
        first_col = row[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1
# Print            
print(counts_dict)


# 3.3
# Work
# Import the pandas package
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize = 10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)
# Check out the head of the DataFrame
print(df_urb_pop.head())
# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode']=='CEB']
# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])
# Turn zip object into list: pops_list
pops_list = list(pops)
# Print pops_list
print(pops_list)


# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)
# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(element[0] * element[1] * .01) for element in pops_list]
# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Concatenate DataFrame chunk to the end of data: data
    data = pd.concat([data, df_pop_ceb])

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()





# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
        
        # Concatenate DataFrame chunk to the end of data: data
        data = pd.concat([data, df_pop_ceb])

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop('ind_pop_data.csv', 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop('ind_pop_data.csv', 'ARB')





