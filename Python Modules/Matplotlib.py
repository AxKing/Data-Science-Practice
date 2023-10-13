# Introduction to Data Visualization with Matplotlib


# Section 1
# Introduction to the pyplot interface
import maplotlib.pyplot as plt

fig, ax = plt.subplots()
# fig is a container
# ax is the canvase

seattle_weather['month']
# this data frame contains 3 letter abreviations for the months

seattle_weather['mly-tavg-normal']
# contains the average temperature, in farenheight, over a 10 year period.

# To add data to the axis, we call a plotting command
ax.plot(seattle_weather['month'], seattle_weather['mly-tavg-normal'])
plt.show()

# could add another layer as well...
ax.plot(austin_weather['month'], austin_weather['mly-tavg-normal'])

# All together
fig, ax = plt.subplots()
ax.plot(seattle_weather['month'], seattle_weather['mly-tavg-normal'])
ax.plot(austin_weather['month'], austin_weather['mly-tavg-normal'])
plt.show()

# Section 2
# Customzing data apperance
fig, ax = plt.subplots()
ax.plot(seattle_weather['month'], seattle_weather['mly-tavg-normal'])
ax.plot(austin_weather['month'], austin_weather['mly-tavg-normal'])
plt.show()

# Our data isn't continuous, so we want to add a marker to the plot
ax.plot(seattle_weather['month'], seattle_weather['mly-tavg-normal'], marker = 'o')
marker = 'v'

# You can change the linestyle
linestyle = "--"
linestyle = 'none' # takes away a line

# You can change colors
color = 'r' #means red

# Label your axis
ax.set_xlabel('Time (months)')
ax.set_ylabel('Temperature')
ax.set_title('title of the plot')
plt.show()


# Section 3 
# Small multiples
fig, ax = plt.subplots()
ax.plot(austin_weather['month'], austin_weather['mly-tavg-normal'], color = 'r')
ax.plot(austin_weather['month'], austin_weather['mly-tavg-25pctl'], linestyle = '--', color = 'r')
ax.plot(austin_weather['month'], austin_weather['mly-tavg-75pctl'], linestyle = '--', color = 'r')
plt.show()

# using Multiple plots or subplots
fig, ax = plt.subplots(3, 2) # this mean 3 rows, and 2 columns

# Create two different subplots. One for Austin, and one for Seattle
fix, ax = plt.subplots(2,1)

ax[0].plot(seattle_weather['month'], seattle_weather['mly-tavg-normal'])
ax[0].plot(seattle_weather['month'], seattle_weather['mly-tavg-25pctl'], linestyle='--', color = 'b')
ax[0].plot(seattle_weather['month'], seattle_weather['mly-tavg-75pctl'], linestyle='--', color = 'b')

ax[1].plot(austin_weather['month'], austin_weather['mly-tavg-normal'], color = 'r')
ax[1].plot(austin_weather['month'], austin_weather['mly-tavg-25pctl'], linestyle = '--', color = 'r')
ax[1].plot(austin_weather['month'], austin_weather['mly-tavg-75pctl'], linestyle = '--', color = 'r')

ax[0].set_ylabel('Preciplitation (inches)')
ax[1].set_ylabel('Preciplitation (inches)')
ax[1].set_xlabel("Time (months)")

plt.show()

fig, ax = plt.subplots(2,1, sharey = True) # this keeps the same range for the y values


# Section 2
# Plotting Time-Series Data
# We need to parse the date column as a date
# climate_chage.index

climate_change = pd.read_csv('climate_change.csv', parse_dates = ['date'], index_col = "date")


import matplotlib.pyplot as plot
fig, ax = plt.subplots()

ax.plot(flimate_change.index, climate_change['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

sixties = climate_change['1960-01-01': '1969-12-31'] # slices the data

fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

sixty_nine = climate_change['1969-01-01':'1969-12-31']
fig, ax =plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time"')
ax.set_ylabel("CO2 (ppm)")
plt.show()

# Section 2  Plotting time-series with different variables
"""
Suppose you have recorded temperature and humidity and CO2 levels across dates
You can plot them both on the sames figure
"""

import pandas as pd
climate_change = pd.read_csv('climate_change.csv', parse_dates=['date'], index_col = 'date')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change['co2'])
ax.plot(climate_change.index, climate_change['relative_temp'])
ax.set_xlabel('Time')
ax.set_ylabel('COS (ppm) / Relative Temperature')
plt.show()

# If the scale if off on both of them, they can be adjusted.
# plotting with two different y axis scales

#First variable
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change['co2'], color = 'blue')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color = 'blue')
ax.tick_params('y', colors ='blue')

ax2 = ax.twinx() # creates a twin set. This means they share the same x, but have different y-axis

ax2.plot(climate_change.index, climate_change['relative_temp'], color = 'red')
ax2.set_ylabel('Relative Temperature (celsius)', color='red')
ax2.tick_params('y', colors='red')
plt.show()


def plot_timeseries(axes, x, y, color, xlabel, ylabel):
	axes.plot(x, y, color=color)
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	axes.tick_params('y', colors=color)

fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time', \
	'Relative Temperature (celsius)')
plt.show()

# Annotating time-series data
ax2.annotate(">1", xy=(pd.Timestamp('2015-10-6'), 1))
# takes in a text, and an xy coordinate
"text, xy = (x,y)"

# positioning
xytext # positioning argument to move the text
ax2.annotate(">1 degree", 
	xy=(pd.Timestamp('2015-10-6'), 1),
	xytext = (pd.Timestamp('2008-10-06'), -0.2)
	arrowprops= {}) # dictionary of properties of an arrow

# arrow properties
arrowprops = {"arrowstyle": "->", "color":"gray"}



# Section 3
# Quantitative comparisons: bar-charts
medals = pd.read_csv('medals_by_country_2016.csv', index_col = 0)
fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
plt.show()

# this will rotate the names of the countries
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')

ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'])

# Adding a second bar
fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
plt.show()

# Adding Bronze Medals
fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'], label = 'Gold')
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'], label = 'Silver')
ax.bar(medals.index, medals['bronze'], bottom=medals['gold']+ medals['silver'], label = 'Bronze')
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
ax.legend()
plt.show()


# Histograms
fig, ax = plt.subplots()
ax.bar("Rowing", mens_rowing['Height'].mean())
ax.bar("Gymnastics", mens_gymnastics['Heights'].mean())
ax.set_ylabel("Heights (cm)")
plt.show()
fig, ax = plt.subplots()


ax.hist(mens_rowing['Height'], label='Rowing', bins = 5, histtype='step') 
# You could also set the bin boarders with a list.
# bins = [150, 160, 170, 180, 190, 200, 210]
ax.hist(mens_gymnastics['Height'], label='Gymnastics', bins = 5)
ax.set_xlabel("Height (cm)")
ax.set_ylabel('# of observations')
ax.legend()
plt.show()

# Statistical Plotting

# Adding error to a bar plot
fig, ax = plt.subplots()
ax.bar("Rowing", mens_rowing["Height"].mean(),\
	yerr=mens_rowing['Height'].std())
ax.bar("Gymnastics", mens_gymnastics['Height'].mean(), \
	yerr=mens_gymnastics['Height'].std())
ax.set_ylabel("Height (cm)")
plt.show()


# Adding error bars to plots
fig, ax = plt.subplots()
ax.plot(seattle_weather['month'], 
	seattle_weather['mly-tavg-normal'],
	yerr = seattle_weather['mly-tavg-stddev']
	)
ax.plot(austin_weather['month'], 
	austin_weather['mly-tavg-normal']
	yerr = austin_weather['mly-tavg-stddev'])
ax.set_ylabel('Temperature (Fahrenheit)')
plt.show()


# Box Plots
fig, ax = plt.subplots()
ax.boxplot([mens_rowing['Height'], mens_gymnastics['Height']]) #pass a list of lists
ax.set_xticklabels(['Rowing', 'Gymnastics'])
ax.set_ylabel('height (cm)')
plt.show()


# Scatter Plot bivariate comparison
fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'])
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()



# Two different scatter data sets on the same.
eighties = climate_change['1980-01-01':'1989-12-31']
nineties = climate_change['1990-01-01':'1999-12-31']


fix, ax = plt.subplots()
# 80's Data
ax.scatter(eighties['co2'], eighties['relative_temp'], color='red', label='eighties')
# 90's Data
ax.scatter(nineties['co2'], nineties['relative_temp'], color='blue', label='nineties')
ax.legend()
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()


fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'],
	c = climate_change.index) # note this is not color.
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()
# Now the color darkness will indicate where in time the measurement was taken





