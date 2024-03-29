# Feature Engineering for Machine Learning in Python

##### Chapter 1
##### Section 1
##### Why Generate Features

# Different Types of Data
# Continuous, categorical, ordinal, boolean, datetime

import pandas as pd
df =pd.read_csv(path_to_csv_file)
print(df.head())

print(df.columns)
print(df.types)

only_ints = df.seclect_dtypes(include=['int'])
print(inly_ints)

##### Chapter 1
##### Section 1
##### Why Generate Features
##### Exercises

# Too Easy

##### Chapter 1
##### Section 2
##### Categorial Features

# Create additional binary features to show if a value was picked or not.

# One-Hot encoding
# Converts n categories into n features.
pd.get_dummies(df,columns=['Country'], prefix='C')

# Dummy encoding
# Converts n categories to n-1 features (leaves off the first one)
pd.get_dummies(df, columns=['Country'], drop_first=True, prefix='C')

counts = df['Country'].value_counts()
print(counts)

mask = df['Country'].isin(counts[count > 5].index)
df['Country'][mask] = 'Other'




##### Chapter 1
##### Section 2
##### Categorial Features
##### Exercises

# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')
# Print the columns names
print(one_hot_encoded.columns)

# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')
# Print the columns names
print(dummy.columns)


# Create a series out of the Country column
countries = so_survey_df.Country
# Get the counts of each category
country_counts = countries.value_counts()
# Print the count values for each category
print(country_counts)

# Create a series out of the Country column
countries = so_survey_df['Country']
# Get the counts of each category
country_counts = countries.value_counts()
# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)
# Print the top 5 rows in the mask series
print(mask.head())

# Create a series out of the Country column
countries = so_survey_df['Country']
# Get the counts of each category
country_counts = countries.value_counts()
# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)
# Label all other categories as Other
countries[mask] = 'Other'
# Print the updated category counts
print(countries.value_counts())



##### Chapter 1
##### Section 3
##### Numeric Variables

# Age, Price, Counts, Geospatial Data
df['Binary_violation'] = 0
df.loc[df['number_of_violations'] > 0, 'Binary_violation'] = 1

# Binning Numberic Variables
import numpy as np
df['Binned_Group'] = pd.cut(
	df['Number_of_violations'],
	bins = [-np.inf, 0, 2, np.inf],
	labels=[1,2,3]
	)


##### Chapter 1
##### Section 3
##### Numeric Variables
##### Exercises

# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0
# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary'] > 0, 'Paid_Job'] = 1
# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())


# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins = 5)
# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())


# Import numpy
import numpy as np
# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]
# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], 
                                         bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())



##### Chapter 2
##### Section 1
##### Dealing with Messy Data

print(df.info())
print(df.isnull())
print(df['StackOverflowJobsRecommend'].isnull().sum())
print(df.notnull())

##### Chapter 2
##### Section 1
##### Dealing with Messy Data
##### Exercises

# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]
# Print the number of non-missing values
print(sub_df.info())

##### Chapter 2
##### Section 2
##### Dealing with missing Values (1)

df.dropna(how='any') # deletes all rows with at least one missing value
df.dropna(subset=['VersionControl']) # drops from these columns
df['VersionControl'].fillna(value='None Given', inplace=True)

df=['SalaryGiven'] = df['ConvertedSalary'].notnull()
df.drop(columns=['ConvertedSalary']) #drops a column

##### Chapter 2
##### Section 2
##### Dealing with missing Values (1)
##### Exercises

# Print the number of rows and columns
print(so_survey_df.shape)

# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna()
# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)


# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())

# Replace missing values
so_survey_df['Gender'].fillna(value='Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())

##### Chapter 2
##### Section 3
##### Dealing with Missing Data (II)

# Replace categorical with most common occuring
# Numeric with mean or median

df['ConvertedSalary'].mean()
df['ConvertedSalary'].median()

df['ConvertedSalary'] = df['ConvertedSalary'].fillna(df['ConvertedSala'].mean())

df['ConvertedSalary'] = df['ConvertedSalary'].astype('int64')

df['ConvertedSalary'] = df['ConvertedSalary'].fillna(round(df['ConvertedSalary'].mean()))

##### Chapter 2
##### Section 3
##### Dealing with Missing Data (II)
##### Exercise

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)
# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)
# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])
# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())


##### Chapter 2
##### Section 4
##### Dealing with other data issues

df['RawSalary'].dtype
df['RawSalary'].head()
df['RawSalary'] = df['RawSalary'].str.replace(',' , '') # remove commas
df['RawSalary'] = df['RawSalary'].astype('float')
df[coerced_vals.isna()].head()

df['column_name'] = df['column_name'].method1().method2().method3()

##### Chapter 2
##### Section 4
##### Dealing with other data issues
##### Exercise

# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')
# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$','')


# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')
# Find the indexes of missing values
idx = numeric_vals.isna()
# Print the relevant rows
print(so_survey_df['RawSalary'][idx])


# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')
# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')
# Print the column
print(so_survey_df['RawSalary'])


# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',','')\
                              .str.replace('$','')\
                              .str.replace('£','')\
                              .astype('float') 
# Print the RawSalary column
print(so_survey_df['RawSalary'])


##### Chapter 3
##### Section 1
##### Conforming to Statistical Assumptions

import matplotlib as plt
# Plot a histogram
df.hist()
plt.show()

# boxplot
df[['column']].boxplot() # takes a list of columns
plt.show()

import seaborn as sns
sns.paiplot(df)

df.describe() # summary of statistics

##### Chapter 3
##### Section 1
##### Conforming to Statistical Assumptions
##### Exercises

# Create a histogram
so_numeric_df.hist()
plt.show()

# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()

# Create a boxplot of ConvertedSalary
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()

# Print summary statistics
print(so_numeric_df.describe())


##### Chapter 3
##### Section 2
##### Scaling and Transformations

# Min-Max scaling
# usually 0-1 on a linear scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['normalized_age'] = scaler.transform(df[['Age']])

# Standardization
# finds the mean and centers the mean around it
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['Age']])
df['standardized_col'] = scaler.transform(df[['Age']])

# Log Transformation
# Transforms
from sklearn.preprocessing import PowerTransformer
log=PowerTransformer()
log.fit(df[['ConvertedSalary']])
df['log_ConvertedSalary'] = log.transform(df[['ConvertedSalary']])


##### Chapter 3
##### Section 2
##### Scaling and Transformations
##### Exercises

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())



# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())



# Import PowerTransformer
from sklearn.preprocessing import PowerTransformer

# Instantiate PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()




##### Chapter 3
##### Section 3
##### Removing Outliers

# Quantile based detection
# top 5%
# To find the 95th percent use the quantile method
q_cutoff = df['col_name'].quantile(0.95)
mask = df['col_name'] < q_cutoff
trimmed_df = df[mask]

# Standard Deviation based Detection
# All data greater than 3 stds around the mean
mean = df['col_name'].mean()
std = df['col_name'].std()
cut_off = std * 3

lower, upper = mean - cutoff, mean + cutoff
new_df = df[(df['col_name'] < upper) & (df['col_name'] > lower)]

##### Chapter 3
##### Section 3
##### Removing Outliers
##### Exercises

# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()



# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) 
                           & (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()


##### Chapter 3
##### Section 4
##### Scaling and Transformations new Data

scaler = StandardScaler()
scaler.fit(train[['col']])
train['scaled_col'] = scaler.transform(train[['col']])

# fit the model
test = pd.read_csv('testcsv.csv')
test['scaled_col'] = scaler.transform(test[['col']])


train_mean = train[['col']].mean()
train_std = train[['col']].std()

cut_off = train_std * 3
train_lower = train_mean - cut_off
train_upper = train_mean + cut_off

#subset train data

test = pd.read_csv('test_data')

test = test[(test[['col']] < train_upper) & (test[['col']] > train_lower)]





##### Chapter 3
##### Section 4
##### Scaling and Transformations
##### Exercises

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())




train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]



##### Chapter 4
##### Section 1
##### Encoding Text

print(speech_df.head())
[a-zA-Z] # all letter characters
[^a-zA-Z] #all non letter characters

speech_df['text'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ') # replace punctuation with spaces

speecf_df['text'] = speech_df['tex'].str.lower() # lower case all of the text

print(speech_df['text'][0])

speech_df['char_cnt'] = speech_df['text'].str.len() # character count or length of all the speeches

speech_df['word_cnt'] = speech_df['text'].str.split() # number of words split on white space

speech_df['word_cnt'].head(1)

speech_df['word_counts'] = speech_df['text'].str.split().str.len()

print(speech_df['word_splits'].head())

speech_df['avg_word_len'] = speech_df['char_cnt'] / speech_df['word_cnt'] #average word length



##### Chapter 4
##### Section 1
##### Encoding Text
##### Exercises

# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')
# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()
# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())

# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()
# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()
# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']
# Print the first 5 rows of these columns
print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])


##### Chapter 4
##### Section 2
##### Word Counts

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
print(cv)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.1, max_df=0.9)

cv.fit(speech_df['text_clean'])
cv_transformed = cv.transform(speech_df['text_clean'])
print(cv_transformed)

cv_transformed.toarray()
feature_names = cv.get_feature_names()

# OR fit and transform
cv_transformed = cv.fit_transform(speech_df['text_clean'])
print(cv_transformed)

# NOw combine them
cv_df = pd.DataFrame(cv_transformed.toarray(), columns = cv.get_feature_names()).add_prefix('Counts_')
print(cv_df.head())

speech_df = pd.concat([speech_df, cv_df], axis=1, sort=False)

##### Chapter 4
##### Section 2
##### Word Counts
##### Exercises

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Instantiate CountVectorizer
cv = CountVectorizer()
# Fit the vectorizer
cv.fit(speech_df['text_clean'])
# Print feature names
print(cv.get_feature_names())


# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])
# Print the full array
cv_array = cv_transformed.toarray()
# Print the shape of cv_array
print(cv_array.shape)


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=0.2, max_df=0.8)
# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()
# Print the array shape
print(cv_array.shape)

# Create a DataFrame with these features
cv_df = pd.DataFrame(cv_array, 
                     columns=cv.get_feature_names()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)
print(speech_df_new.head())


##### Chapter 4
##### Section 3
##### Term Frequency-Inverse Document frequency

print(speech_df['Counts_the'].head())

from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
print(tv)

tv.fit(train_speech_df['text'])
train_tv_transformed = tv.transform(train_speech_df['text'])

# Combine the TF-IDF values alone with the feature names
train_tv_df = pd.DataFrame(train_tv_transformed.toarray(), columns=tv.get_feature_names()).add_prefix('TFIDF_')

train_speech_df = pd.concat([train_speech_df, train_tv_df], axis=1, sort=False)

examine_row = train_tv_df.iloc[0]
print(examine_row.sort_values(ascending=False))


test_tv_transformed = tv.transform(test_df['text_clean'])

test_tv_df = pd.DataFrame(text_tv_transformed.toarray(), columns=tv.get_feature_names()).add_prefix('TFIDF')

test_speech_df = pd.concat([test_speech_df, test_tv_df], axis=1, sort=False)


##### Chapter 4
##### Section 3
##### Term Frequency-Inverse Document frequency
##### Exercises


# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')
# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])
# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), 
                     columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())


# Isolate the row to be examined
sample_row = tv_df.iloc[0]
# Print the top 5 words of the sorted output
print(sample_row.sort_values(ascending=False).head(5))

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')
# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])
# Transform test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])
# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), 
                          columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(test_tv_df.head())



##### Chapter 4
##### Section 4
##### N-Grams

tv_bi_gram_vec = TfidfVectorizer(ngram_range = (2,2))

# fit and apply bigram vectorizer
tv_bi_gram = tv_bi_gram_vec.fit_transform(speech_df['text'])

# Print the bigram features
print(tv_bi_gram_vec.get_feature_names())

tv_df = pd.DataFrame(tv_bi_gram.toarray(), columns=tv_bi_gram_vec.get_feature_names()).add_prefix('Counts_')

tv_sums = tv_df.sum()
print(tv_sums.head())

print(tv_sums.sort_values(ascending=False)).head()



##### Chapter 4
##### Section 4
##### N-Grams
##### Exercises

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100, 
                                 stop_words='english', 
                                 ngram_range=(3,3))
# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])
# Print the trigram features
print(cv_trigram_vec.get_feature_names())
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_trigram.toarray(), 
                 columns=cv_trigram_vec.get_feature_names()).add_prefix('Counts_')
# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending=False).head())



