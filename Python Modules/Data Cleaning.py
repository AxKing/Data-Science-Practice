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













