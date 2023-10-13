"""
Statistical inference
Confidence Intervals
Bootstrapping
Hypothesis Testing
P Values and Confidence Intervals

Statistical Modeling
Fitting Models to Data
Goodness of Fit
Cross Validation
Logistic Regression
Baysian Inference
"""
import math
import io

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as pp
%matplotlib inline

import scipy.stats
import scipy.optimize
import scipy.spatial

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


poll.vote.value_counts(normalize=True)

np.random.rand(5) < .51
# gives a number

def sample(brown, n=1000):
	return pd.DataFrame({'vote': np.where(np.random.rand(n) < brown, 'Brown', 'Green')})

n = sample( .51, n=1000)
dist = pd.DaraFrame({sample(0.51).vote.value_counts[normalize=True] for i in range(1000)})
dist.head()

dist.Brown.hist(histtype='step', bins=20)


# Confidence interval
"""XX% confidence interval:
* Built from the data

* Contains the true value of a population parameter
XX% of the time

* Over many similar experiments

"""


def samplingdist(brown, n=1000):
	return pd.DataFrame([sample(brown,n).vote.value_counts(normalize=True) for i in range(1000)])

def quantiles(brown, n=1000):
	dist = samplingdist(brown,n)
	return dist.Brown.quantile(0.025), dist.Brown.quantile(0.975)

quantiles(0.5)
quntiles(0.48)
quantiles(0.54)

# 95% confidence interval: [0.48,0.54]
# Estimate = 0.51 +- 0.003 (at 95% confidence)

# Now we'll create a new distribution with .5 change of probability and 10000 poles
dist = samplingdist(0.50, 10000)
dist.Brown.hist(histtype='step')



# Bootstrapping
pop = pd.read_csv('grades.csv')
pop.grade.hist(histtype='step')

pop.describe()
"""
Bootstrapping
What we'll do is to estimate the uncertainty of our statistic, 
the mean, by generating a large family of samples from the one we have.
And then chracterizing the distribution of the mean over this family.

Each sample in the family is prepared as follows:
- We draw grades randomly for our single esisting sample, 
allowing the same grade to be drawn more than once.
This is sampling with replacement

"""
pop.sample(100, replacement=True).describe()

bootstrap = pd.DaraFrame({'meangrade': [pop.sample(100, replacement=True).grade.mean() for i in range(1000)]})

bootstrap.meangrade.hist(histtype='step')
pp.axvline[pop.grade.mean().color='C1']

bootstrap.meangrade.quantile(0.025), bootstrap.meangrade.quantile(0.975)

n1 = scipy.stats.norm(7.5,1)
n2 = scipy.stats.norm(4,1)

x = np.linespace(0,10,100)
pp.plot(x,0.5*n1.pdf(x) + 0.5*n2.pdf(x))

def draw():
	while True:
		v = nl.rvs() if np.random.rand() < 0.5 else n2.rvs()
		if v <= v <= 10:
			return v

def dataset(n=100):
	return pd.DataFrame({'grade':[draw() for i in range(n)]})

for i in range(5):
	dataset(100).grade.hist(histtype='step', density=True)

means = pd.DataFrame({'meangrade': [dataset(100).grade.mean() for i in range(1000) ] })

means.meangrade.hist(histtype='step')
bootstrap.meansgrade.hist(histtype='step')



# Hypothesis Testing
"""

Observe statistic in data
Compute sampling distribution of statistic under null hypothesis
Quantile of observed statistic gives P value

"""


pumps = pd.read_csv('pumps.csv')
cholera = pd.read_csv('cholera.csv')

cholera.loc[0::20] # looks at every 20th record
pp.figure(figsize=[6,6])
pp.scatter(pumps.x, pumps.y, color='b')
pp.scatter(cholera.x, cholera.y, color = 'r', s=3)

img = matplotlib.image.imread('london.png')
pp.figure(figsize=[10,10])
pp.imshow(img, extent=[-0.38, 0.38, -0.38, 0.38])

cholera.closest.value_counts()

cholera.groupby('closest').deaths.sum()

def simulation(n):
	return pd.DataFrame({'closest': np.random.choice[[0,1,4,5], size=n,p=[0.65, 0.15, 0.1, 0.1]]})

simulate(489).closest.value_counts()

sampling = pd.DataFrame({'counts' : [simulate(489).closest.value_counts()[0] for i in range(10000)]})
sampling.counts(hist(histype='step'))

scipy.stats.percentileofscore(sampling.counts,340)
# returns 98.14

1 - 98.14
1.859999

# We would only expect a count of 340 deaths 1.859% of the times

# This is the P-Value. The smaller the p-value, the more strongly we can reject the null hypothesis.



# P Values and Confidence Intervals
"""
There is a close relationship between hyptheis testing and confidence intervals.
If the null hypothesis cooresponds to a range of values for a population parameter that
are excluded from the confidence interval the Null Hypothesis must be rejected 
"""

poll  = pd.read_csv('poll.csv')
poll.vote.value_counts(normalize=True)

def sample(brown, n=1000):
	return pd.DataFrame({'vote': np.where[np.random.rand(n) < brown, 'Brown', 'Green']})

dist = pd.DataFrame({'votes':[sample(0.5, 1000).vote.value_counts(normalize=True)['Brown'] for i in range(10000)]})

dist.Brown.hist(Brown.hist[histtype='step', bins=20])

100 - scipy.stats.percentileofscore(dist.Brown, 0.511)
# came out to 24.39

largepoll - pd.read_csv('poll_larger_data.csv')
largepoll.vote.value_counts(normalize=True)

dist = pd.DataFrame({'Green': [sample(0.5, 10000).vote.value_counts(normalize=True)['Green'] for i in range(1000) ]})

dist.Green.hist(histtype='step,bins=20')
pp.axvline(0.51, c='C1')




# Statistical Modeling
# May releave facts and trends about population
# predict future behavior

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

gapminder = pd.read_csv('gapminder.csv')
gdata = gapminder.query('year == 1985')

size = 1e-6 * gdata.population
colors = gdata.region.map({'Africa': 'skyblue', 'Europe': 'gold', 
	'America': 'palegreen', 'Asia': 'coral'})
def plotdata():
	gdata.plot.scatter('age5_surviving', 'babies_per_woman', 
			c= colors, s=size, linewidths = 0.5, edgecolor='k', alpha=0.5)

plotdata()

# stats models as smf
# OLS = Ordinary Least Squares (How we'll find the coefficients for the model)
smf.ols(formula='babies_per_woman - 1', data=gdate)
grandmean = model.fit()
grandmean

def plotfit(fit):
	plotdata()
	pp.scatter(gdata.age5_surviving, fit.predict(gdata)
		, c=colors, s=30, linewidth=0.5, edgecolor ='k', marker ='D')

plotfit(grandmean)

grandmean.params
gdata.babies_per_woman.mean()

groupmeans = smf.ols(formula='babies_per_woman - 1 + region', data=gdate).fit()
plotfit(groupmeans)
groupmeans.params



# Goodness of fit
"""
Measurements for the Goodness of Fit
- Mean Squared Error
	This is a measurement of the average error value of each data point.
	Average the square of the distances from the model to the actual data points
- R^2 = (Explained variance)/(Total Variance)
	Explained Variance = Variance of the model
	Total Variance = Variance of the Data Set
	Maximum R^2 is 1
	Minimum is 0
- F statistic: explanatory power of fit parameters compared to "random" fit vectors
	Measures how each parameter influences R^2 compared to another random vector
	A value of 1 means the parameter influences R^2 the same as a random vector 
	The larger the value, the greater the influence on R^2
- ANOVA Table- Analysis of Variance
	df region - degrees of freedom
	df Residual - number of data points - number of parameters
	sum_sq, mean_sq - errors that we've calcuted already
	F is F statistic
	PR(>F) is the P value of the model 
"""
# MSE
# Mean Squared Error for the model
model.mse_resid 

# R^2
model.rsquared

# F Statistic
model.fvalue


sm.stats.anova_lm(groupmeans)


# Cross Validation
""" 
Type of validation that is used to compare models in machine learning
Data is divided into a training data set and a testing data set
"""

def cross_validate(data, formula, response, n=100):
	ret = []

	for i in range(n):
		shuffled = data.sample(len(data))
		# Divide the data in 2
		training, testing = shuffled.iloc[:len(data)//2], shuffled.iloc[len(data)//2:]
		# train the data
		trained = smf.ols(formula, data = training).fit()

		resid = trained.predict(testing) - testing[response]
		df = len(testing) - trained.df_model - 1
		mse = np.sum(resid**2) / df

		ret.append(mse)

	return np.mean(ret)



# Logistic Regression
""" 
Takes a standard linear model and converts it into something bound by 0 and 1
e^x/(1+e^x) is always between 0 and 1

Cannot use least squres for this
"""
smf.logit()


# Bayseian Inference
"""
* We do not make estimates of population parameters from the data
* Rather, we maintain probability distributions for population distributions, which
represent our quantitative belief about their value
* We start with probability priors, and we use observations to update them 
to probability posteriors
"""

import pymc3 as pm








