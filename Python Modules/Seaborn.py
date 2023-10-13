# Seaborn

import seaborn as sns
import matplotlib.pyplot as plt

# scatterplot
sns.scatterplot(x=height,y=weight)
plt.show()

# Count Plot
# Take in lists of data nad plot bars for each category of data
sns.countplot(y=gender)
plt.show()

import pandas as pd
df = pd.read_csv('masculinity.csv')
df.head()

# countplot with a dataframe
sns.countplot(x='how_masculine', data=df)
plt.show()

# adding color to our graphs
tips = sns.load_dataset('tips')
tips.head()


hue_colors = {"yes":"black", "no":"red"}

sns.scatterplot(x='total_bill', y='tip',data=tips, hue='smoker', hue_order=['yes', 'no'], palette=hue_colors)
plt.show()

# You can use hexcodes to make your own colors
hue_colors = {"yes": '#808080',
				"no": "#00FF00"
}


# Relational plots
# Plots and Sub plots
relplot()

sns.scatterplot(x='total_bill', y='tip',data=tips)
# These are the same
sns.relplot(x='total_bill', y='tip',data=tips, kind="scatter")

# To arange subplots use col=
sns.relplot(x='total_bill', y='tip',data=tips, kind="scatter", col='smoker')
# to range them vertically use row=
sns.relplot(x='total_bill', y='tip',data=tips, kind="scatter", row='smoker')
# to use both...
sns.relplot(x='total_bill', y='tip',data=tips, kind="scatter", col='smoker', row='time')

col_wrap=2 #is an optional argument saying how many displays you want per column
sns.relplot(x='total_bill', y='tip',data=tips, kind="scatter", col='day', col_wrap=2, col_order=['Thur', 'Fri', 'Sat', 'Sun'])


# Customizing
# we have seen col, row, hue
# point size, style, and transparency

# size
size = 'size'
hue = 'size'

# point style 
hue = 'smoker'
style = 'smoker'

# transparency
# alpha is a value that is set between 0 and 1
alpha = 0.4

# line plots
sns.relplot(x='hour',y='NO_2_mean', data=air_df_loc_mean, 
	kind='line', style='location', hue='location', markers=True)

# With markers set, the lines will be made of markers too
# set dashes=False to make the lines solid again
dashes=False

# built in 95% confidence interval
sns.relplot(x='hour', y='NO_2', data=air_df, kind='line')

# To use standard deviation, set ci='sd'
sns.relplot(x='hour', y='NO_2', data=air_df, kind='line', ci='sd')

# Turn off the confidence interval by ci=None



# Categorical plots catplot
# Count plots and bar plots

catplot()
sns.catplot(x='how_masculine', data=masculinity_data, kind='count')
plt.show()

# to change the order of categories, make a list
category_order = ['No answer', 'not at all', 'not very', 'somewhat', 'very']
sns.catplot(x='how masculine', data=masculinity_data, kind='count', order=category_order)

# bar plots will show the mean of observations in each category
sns.catplot(x='day', y='total_bill', data=tips, kind='bar')
ci=None # will get rid of the confidence interval


# Box Plots
g = sns.catplot(x='time', y='total_bill', data=tips, kind='box')
# order = []
sym='' #omit outliers
#whiskers default to 1.5 * IQR
whis = 2.0
whis = [5,95] #5th percentile, 95th percentile
whis = [0,100] #set these to the min and max values



# Point Plots 
sns.catplot(x='age',y='masculinity_important', hue='feel_masculine', kind='point')

# to remove the lines connecting the points
join = False

from numpy import median
# for median
esimator = median

# to show caps at the end of the intervals
capsize = 0.2

# turn off the confidence interval 
ci = None







