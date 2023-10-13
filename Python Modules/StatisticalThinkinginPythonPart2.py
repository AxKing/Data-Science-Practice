# StatisticalThinkinginPythonPart2

# Section 1 Optimal Parameters
"""
Optimal Parameters are parameter values that bring the model in closest agreement with the data.
"""
# To check the speed of light estimation, we assumed
import numpy as np
import matplotlib.pyplot as plt
mean = np.mean(michelson_speed_of_light)
std = np.std(michelson_speed_of_light)
samples = np.random.normal(mean, std, size=10000)


# Examples
# Seed random number generator
seed = np.random.seed(42)
# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)
# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)
# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')
# Show the plot
plt.show()


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    return x, y




# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)
# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)
# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
# Margins and axis labels
plt.margins(.02)  # <- 2% error margins
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')
# Show the plot
plt.show()


# Section 2 Linear Regression by Least Squares
"""
Least Squares
The Process of finding the parameters for which the sum of the squares of the residuals is minimal

Also called minimizing RSS, Residual Sum of Squares

Finding the parameters
"""
# np.polyfit() is the function we're going to use to estimate the line
slope, intercept = np.polyfit(total_votes, dem_share, 1)
# this is x and y data and the degree of the polynomial we want to fit.



def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]
# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_width, versicolor_petal_length)

# Print the result
print(r)



# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, 1)
# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')
# Make theoretical line to plot
x = np.array([0,100])
y = a * x + b
# Add regression line to your plot
_ = plt.plot(x, y)
# Draw the plot
plt.show()



# Specify slopes to consider: a_vals
a_vals = np.linspace(0,0.1, 200) # creates a blank array of 100 values between 0 and 0.1
# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals) # creates an empty copy of the same shape as a_vals
# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)
# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')
plt.show()



# Section 3 The importance of EDA: Anscombe's quartet

# Perform linear regression: a, b
a, b = np.polyfit(x,y, 1)
# Print the slope and intercept
print(a, b)
# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = x_theor * a + b
# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(x_theor, y_theor)
# Label the axes
plt.xlabel('x')
plt.ylabel('y')
# Show the plot
plt.show()


# Chapter 2 Generating Bootstrap replicates
"""
Bootstrapping- the use of resampled data to perform statistical inference

Bootstrap sample- a resampled array of the data

Bootstrap replicate- A statistic computed from a resampled array


"""
np.random.choice(samples, size=n) #This takes in an array and a size

import numpy as numpy
np.random.choice([1,2,3,4,5], size = 5)

bs_sample = np.random.choice(michelson_speed_of_light, size=100)

np.mean(bs_sample)
np.median(bs_sample)
np.std(bs_sample)



# Bootstrapping rain fall data
for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)
# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')
# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
# Show the plot
plt.show()



# Bootstrapping Confidence intervals
def bootstrap_replicate_1d(data, func):
	"""Generate bootstrap replicate of 1D data. """
	bs_sample = np.random.choice(data, len(data))
	return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size=size)
    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates



bootstrap_replicate_1d(michelson_speed_of_light, np.mean)

# Take a bunch of bootstrap replicates
bs_replicates = np.empty(10000)
for i in range(10000):
	bs_replicates[i] = bootstrap_replicate_1d(michelson_speed_of_light, np.mean)

# Plot the replicates
_ = plt.hist(bs_replicates, bins=30, normed=True) 
# normed = True makes the bars of the histogram between 0 and 1, mimicking a probability density function
_=plt.xlabel('mean speed of light (mk/s)')
_=plt.ylabel('PDF')
plt.show()

"""
If we repeated measurements over and over again, 
p% of the observed values would lie within the p% confidence interval
"""

conf_int = np.percentile(bs_replicates, [2.5,97.5]) # 95% confidence interval





# Bootstrap example using functions defined up above
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)
# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))  #This is the SEM for the actual data set
print(sem)
# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
# sem and bs_std should be equal
print(bs_std)
# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')
# Show the plot
plt.show()
# 95% confidence interval fo the mean
np.percentile(bs_replicates, [2.5,97.5])
# returned 779.7699, 820,9504


# Exploring the variance 
# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, 10000)
# Put the variance in units of square centimeters
bs_replicates = bs_replicates / 100.0
# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')
# Show the plot
plt.show()


# Back to our baseball example and using bootstrapping 
# to create an interval of when this will happen again

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, 10000)
# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5,97.5])
# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')
# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')
# Show the plot
plt.show()


# Nonparametric inference
# nonparametric- Make no assumptions about the model of probability distribution underlying the data

# We can get estimates for our slope and intercept using bootstrapping

# Pairs Bootstrap for linear regression
# Resample data in pairs
# Compute slope and intercept from resampled data
# Each slope and intercept is a bootstrap replicate
# Compute confidence intervals from percentiles of bootstrap replicates

np.arange(n) #returns an array [0,1,2,...,n-1]
inds = np.arange(len(total_votes))
bs_inds = np.random.choice(inds, len(inds))
# use this random indicie to grab data from a pair of arrays
bs_total_votes = total_votes[bs_inds]
bs_dem_share = dem_share[bs_inds]

bs_slope, bs_intercept = np.polyfit(bs_total_votes, bs_dem_share, 1)
# compared to the acctual value
np.polyfit(total_votes, dem_share, 1)


def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps



# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, 1000)
# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5,97.5]))
# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()


# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])
# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')
# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()


# Chapter 3 Formulating and simulating a hypothesis

"""
How do we assess how reasonable it is that our observed data rae actually described by the model?

Hypothesis testing - an assessment of how reasonable the 
ovserved data are assuming a hypothesis is true

The hypothesis we are testing is called the Null Hypothesis

Permutation - Random reordering of entries in an array
"""

import numpy as np
dem_share_both = np.concatenate((dem_share_PA, dem_share_OH)) # Takes in a touple
dem_share_perm = np.random.permutation(dem_share_both) # mix them up
perm_sample_PA = dem_share_perm[:len(dem_share_PA)] # These are permutation samples
perm_sample_OH = dem_share_perm[len(dem_share_PA):] # These are permutation samples


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))
    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


# Test Statistics and p-values

"""
# Test Statistic - A single number that can be computed 
# from observed data and from data you simulate under the null hypothesis

# It serves as a basis of comparison between what the hypothesis predicts and what we can observe
"""

# Are the two states different?
# picking the difference in means

np.mean(perm_sample_PA) - np.mean(perm_sample_OH)
# permutation replicate - a test statistic generated from a permutation

np.mean(dem_share_PA) - np.mean(dem_share_OH) #origonal data

"""
After we retest thousands of simulations of replicates, we will plot the differences of means
on a histogram


The vast majority of the values were between -4 and 4 %
Actual mean difference was 1.6%
The area to the right of the red line says that at least 23% of the elections had a 1.6% difference
or greater.
That percentage, .23 is called a p-value

P-Value - The probability of obtaining a value of your test statistic that is at least as extreme 
as what was obvserved, under the assumption the null hypothesis is true.

The P-value is NOT the probability that the null hypothesis is true

Statistical Significance
- determined by the smallness of P

NHST is Null Hypothesis Significance Testing (NHST)

"""
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates


# We're going to look at an example of two frogs that were chasing flies
# The Null hypothesis, "There is no difference in the impact of force between frogs"

# Make bee swarm plot to see the data between the two frogs
_ = sns.swarmplot(x='ID', y='impact_force', data = df)
# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')
# Show the plot
plt.show()


# this is us testing our frog data 
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = np.mean(force_a) - np.mean(force_b) # Actual force
# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)
# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
# Print the result
print('p-value =', p)

# The p-value is 0.6%
# A P-value belw 0.01 is stastically significant
"""
# Section 3 Bootstrap hypothesis tests
Pipeline for hypothesis testing
- Clearly state the null hypothesis
- Generate many sets of simulated data assuming the null hypothesis is true
- The p-value is the fraction of your simulated data sets for which the test statistic is at least
as extreme as the real data.
"""

"""
Michelson and Newcomb were both working on calculating the speed of light.
Michelson collected numerous data points.
For Newcomb, we only have his mean.

Could Michaelson gotten the data set that he did if the true mean for the experiments was 
Newcombs mean? 

Null Hypothesis - The true mean speed of light in Michelson's experiments was actually 
Newcomb's reported value.

Because we're comparing a with a data set, a permutation test is not applicable.

We shift over the data with the new mean speed of light. 

"""
newcomb_value = 299860 #km/s
michelson_shifted = michelson_speed_of_light - np.mean(michelson_speed_of_light) + newcomb_value

def diff_from_newcome(data, newcomb_value=299860):
	return np.mean(data) - newcomb_value

diff_obs = diff_from_newcomb(michelson_speed_of_light)
diff_obs

# The test statistic is the mean minus newcomb's value.
# We wrote a function for that.

# Now we'll use the function we already wrote
# then we will calculate the value of the test statistic with a bootstrapped sample on the shifted data
bs_replicates = draw_bs_reps(michelson_shifted, diff_from_newcomb, 10000)
p_value = np.sum(bs.bs_replicates <= diff_observed)/10000





# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + .55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)



# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000 
print('p-value =', p)




"""
# Chapter 4
A/B testing

The null hypothesis of an A/B test is that the statistic is impervious to the change.
"""
import numpy as np
# clickthrough_A, clickthrough_B is an array of 0 and 1 for click throughs
def diff_frac(data_A, data_B):
	frac_A = np.sum(data_A) / len(data_A)
	frac_B = np.sum(data_B) / len(data_B)
	return frac_B - frac_A

diff_frac_obs = diff_frac(clickthrough_A, clickthrough_B)

# Now we can generate 10000 replicates
perm_replicates = np.empty(10000)
for i in range(10000):
	perm_replicates[i] = permutation_replicate(clickthrough_A, clickthrough_B, diff_frac)

# We compute the p-value as the number of replicates where the 
# test statistic was at least as great as what we observed.

p_value = np.sum(perm_replicates >= diff_frac_obs)/10000
print(p_value)



# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, 10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)




# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs)/len(perm_replicates)
print('p-val =', p)



# Test of correlation
"""
Posit null hypothesis: The two variables are completly uncorrelated

Simulate data assuming null hypothesis is true

Use Pearson correlation, p, as test statistic

Compute p-value as fraction of replicates that have p at least as large as observed.


"""


# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs)/10000
print('p-val =', p)



# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()




# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)
# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))
# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count
# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)
# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated
# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)





"""
Final Prject about finches
"""
# Create bee swarm plot
_ = sns.swarmplot(x='year', y='beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()



# Chapter 5


# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)
# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')
# Set margins
plt.margins(0.02)
# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')
# Show the plot
plt.show()



# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)
# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)
# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975
# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5,97.5])
# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')



# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))
# Shift the samples
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean
# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, 10000)
# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975
# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)
# Print p-value
print('p =', p)



# draw_bs_pairs_linreg()



"""
In the second section, we're going to look at beak depth and beak length together."""


# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color = 'blue', alpha=0.5)
# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)
# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')
# Show the plot
plt.show()




# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)
# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = \
        draw_bs_pairs_linreg(bl_1975, bd_1975, 1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = \
        draw_bs_pairs_linreg(bl_2012, bd_2012, 1000)
# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5,97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5,97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5,97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5,97.5])
# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)



# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)
# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)
# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')
# Generate x-values for bootstrap lines: x
x = np.array([10, 17])
# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')
# Draw the plot again
plt.show()




# Compute length-to-depth ratios
ratio_1975 = bl_1975/bd_1975
ratio_2012 = bl_2012/bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, 10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)



# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)
# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')
# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')
# Show plot
plt.show()



def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[i], y[i]
        bs_replicates[i] = func(bs_x, bs_y)
    return bs_replicates





# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)
# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, 1000)
bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, 1000)
# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5,97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5,97.5])
# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)






# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)
# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted, bd_offspring_scandens)
# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)
# Print the p-value
print('p-val =', p)
