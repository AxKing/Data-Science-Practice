# Statistics in Python

# Chapter 1
# Introduction to Exploratory Data Analysis
import pandas as pd
df_swing = pd.read_csv('2008_swing_states.csv')
df_swing[['state','county', 'dem_share']]

# Plotting a histogram
import matplotlibb.pyplot as plt
_ = plt.his(df_swing['dem_share'])
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
plt.show()

# customizing where your bins are
bin_edges = [0,10,20,30,40,50,60,70,80,90,100]
_ = plt.hist(df_wing['dem_share'], bins=bin_edges)
plt.show()

# OR have them evenly spaced out
_ = plt.hist(df_swing['dem_share'], bins=20)
plt.show()

# Seaborn
import seaborn as sns

sns.set()
_ = plt.hist(df_swing['dem_share'])
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
plt.show()


# Bee Swarm Plot
# Give the graph which values will be the x-axis, and the y-axis, and the name of a data frame data
_ = sns.swarmplot(x='state', y='dem_share', data =df_swing)
_ = plt.xlabel('state')
- = plt.ylabel('percent of vote for Obeezile')
plt.show()


# Empirical Cumulative Distribution Function (ECDF)
# X-axis is sorted data
x = np.sort(df_swing['dem_share']) # this data must be sorted
y = np.arange(1, len(x)+1) / len(x) 
_ = plt.plot(x, y, marker='.' , linestyle='none')
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
plt.margins(0.02)
plt.show()



# Chapter 2
# Introduction to summary statistics: The sample Mean and Median
import numpy as np
np.mean(dem_share_PA)
# The mean is heavily influenced by outliers
# The median is the middle value of a data set, it is immune to outlires
np.median() # the 50th percentile

np.percentile(df_swing['dem_share'], [25,50,75]) # pass in a set of data and the percentiles you want

# Box Plots or Box and Whisker Plot
import matplotlib.pyplot as plt
import seaborn as sns

_ = sns.boxplot(x='east_west', y='dem_share', data = df_all_states)
_ = plt.xlabel('region')
_ = plt.ylabel('percent of vote for Obama')
plt.show()

# calculating variance
np.var()

# standard deviation
np.std()

# Scatter Plot
_ = plt.plot(total_votes/1000, dem_share, marker = '.', linestyle ='none')
_ = plt.xlabel('total votes (thousands)')
_ = plt.ylabel('percent of vote for Obama')

# Covariance
 np.cov() 

# Random number generatos and hacker statistics
np.random.random() # draws between 0,1
# Bernulli trials are true or false.

np.random.seed() #integer fed into random number generating algorithm
np.random.seed() # is a sudo random number generator

# Simulatig 4 coin flips
import numpy as np
np.random.seed(42)
random_numbers = np.random.random(size=4)
print(random_numbers)
heads = random_numbers < 0.5
print(heads)
np.sum(heads)

n_all_heads = 0 # Initialize number of 4-heads trials
for x in range(10000):
	heads = np.random.random(size=4) < .5
	n_heads = np.sum(heads)
	if n_heads == 4:
		n_all_heads += 1

n_all_heads/10000

# Discrete Uniform PMF
# distribution is a mathematical description of outcomes
# discrete
# uniformly distributer

np.random.binomial(4, 0.5)

np.random.binomial(4,0.5,size=10)

samples = np.random.binomial(60, 0.1, size=10000)
n = 60
p = 0.1

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
x, y = ecdf(samples)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('number of successes')
_ = plt.ylabel('CDF')
plt.show()




# Poisson process
samples = np.random.poisson(6,size=10000)
x , y = ecdf(samples)
_ = plt.plot(x,y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Number of successes')
_ = plt.ylabel('CDF')
plt.show()

 
# Probability Density Functions
# Normal distribution
import numpy as numpymean  
mean = np.mean(michelson_speed_of_light)
std = np.std(michelson_speed_of_light)
samples = np.random.normal(mean, std, size=10000)
x,y = ecdf(michelson_speed_of_light)
x_theor, y_theor = ecdf(samples)

import mayplotlib.pyplot as plot
import seaborn as sns
sns.set()
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('speed of light (km/s)')
_ = plt.ylabel('CDF')
plt.show()



np.random.normal(mean, std, size=10000)

# Exponential Distribution
mean = np.mean(inter_times)
samples = np.random.exponential(mean, size=10000) # Exponential distribution
x, y = ecdf(inter_times)
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x,y, marker='.', linestyle='none')
_ = plt.xlabel('time (days)')
_ = plt.ylabel('CDF')
plt.show()




def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0][1] / covariance_matrix[0][0]
# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)
# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)
        
replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)

# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5,97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5,97.5])
# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)



