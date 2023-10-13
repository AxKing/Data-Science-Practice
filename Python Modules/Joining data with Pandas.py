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



.merge() 								VS 			.merge_ordered()
* Coulums(s) to join on each table 					* Coulums(s) to join on each table
	on, left_on, right_on 								on, left_on, right_on 					
* Type of join 										* Type of join
	how (left, right, inner, outer)   					how (left, right, inner, outer)
	default to inner 									default to outer
* Overlapping column names							* Overlapping column names
	suffixes = [] 											suffixes =[]
*calling the method   								*calling the method
	df1.merge(df2)										pd.merge_ordered(df1,df2)
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
variablevalue relationships.
If you want to keep column A, and D after the melt
B and C will be in the Variable column and their values will coorespond to the value column

"""

social_fin_tall = social_fin.melt(id_vars=['financial','company'], value_vars=['2018', '2017'] \
					var_name=['year'], value_name = 'dollars')













