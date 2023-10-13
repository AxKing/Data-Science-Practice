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








