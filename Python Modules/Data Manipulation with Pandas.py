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
index	list	cumsum
0 		24		24
1 		24		48
2 		24 		72
3   	17 		89		
4 		29 		118
5 		2 		120
6 		74   	194


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















