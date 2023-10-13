"""
Importing Data from a ton of different files.
# plain text files
# Reading a flat file or csv using NumPy or pandas
# importing excel spreadsheets
# pickled files
# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA
# Importing HDF5 files
# Hierarchical Data Format 5
# MATLAB and .mat files
# SQL Queries in Python
# Webdata from a URL
# webscraping using requests and urllib
# scraping using beautiful soup
# APIs (Application Programming Interface)
# JSONs (JavaScript Object Notation)
# tweepy 
# html parsing
"""

Importing data from a variety of sources

plain text files

table data

# reading a text or txt file
filename = 'fuck_finn.txt'
file = open(filename, mode='r') #r is to read(
text = file.read()
file.close()

print(file)
with open('huck_finn.txt', 'r') as file:  #context manager
	print(file.read())


# Open a file: file
file = open('moby_dick.txt', 'r')
print(file.read())
# Print it
print(file)
# Check whether file is closed
print(file.closed)
# Close file
file.close()
# Check whether file is closed
print(file.closed)

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

# Reading a flat file or csv
# NumPy or pandas
loadtxt() #breaks down when there is numeric data
import numpy as np
filename = 'MNIST_Header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols[0,2], dtype=str) 
# delimiter defaults to white space
# skiprows is in case you have a header
# usecols is what columns of the daya you want to use.
# dtype=str
print(data)

# Import package
import numpy as np
# Assign filename to variable: file
file = 'digits.csv'
# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')
# Print datatype of digits
print(type(digits))
# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))
# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()


# Import numpy
import numpy as np
# Assign the filename: file
file = 'digits_header.txt'
# Load the data: data
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2])
# '\t' is tab delimited
# skip the first row
# we grabbed the first and 3rd columns


# Assign filename: file
file = 'seaslug.txt'
# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)
# Print the first element of data
print(data[0])
# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)
# Print the 10th element of data_float
print(data_float[9])
# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()


data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
# first is filename
# second is comma delimeter
# names asks if there is a header
# dtype asks if we want to specify a data type or let the function figure it out.

#read from a csv
np.recfromcsv() # behaves similarly to read from csv and read from text.
# defaults to delimiter = ','
# defalts to names = True
# defaults to dtype = None

#importing flat files using pandas
import pandas as pd
filename = 'winequality-red.csv'
data = pd.read_csv(filename)
data.head()


# Assign the filename: file
file = 'digits.csv'
# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header= None)
# Build a numpy array from the DataFrame: data_array
data_array = np.array(data)
# Print the datatype of data_array to the shell
print(type(data_array))

# importing a bad file from pandas
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Assign filename: file
file = 'titanic_corrupt.txt'
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values=['NA', 'NaN', 'Nothing'])
# sep #separator
# comment is the character for commenting
# na_values is a list of strings that are null

# Print the head of the DataFrame
print(data.head())
# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()

# pickled files
# many data types
import pickle
with open('pickled_fruit.pkl', 'rb') as file: 
#rb is read only in binary
	data = pickle.load(file)
print(data)

#importing excel spreadsheets
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)

df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float you want to load as a data frame



# Import pandas
import pandas as pd
# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xls
xls = pd.ExcelFile(file)
# Print sheet names
print(xls.sheet_names)
# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')
# Print the head of the DataFrame df1
print(df1.head())
# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)
# Print the head of the DataFrame df2
print(df2.head())




# Importing SAS and Sata files using Pandas
# SAS - Statistical Analysis System
# Stata - STAtistics and daTA

# common SAS files are Cat and Dat which are catalogue and Data Set files
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sass7bdat') as file:
	df_sas = file.to_data_frame()

# Example
# Import sas7bdat package
from sas7bdat import SAS7BDAT
# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
# Print head of DataFrame
print(df_sas.head())
# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()



#importing stata files
import pandas as pd
data = pd.read_stata('urbanpop.dta')

# Import pandas
import pandas as pd
# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')
# Print the head of the DataFrame df
print(df.head())
# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()


# Importing HDF5 files
# Hierarchical Data Format 5
# Standard for storing large quantities of numerical data

import h5py
filename = 'whatever'
data = h5py.File(filename, 'r') #r is to read
print(type(data))

for key in data.keys():
	print(key)

"""meta- meta data for the file
quality- referes to data quality.
strain- strain data from the interferometer. In some sense, this is "the data"""

for key in data['meta'].keys():
	print(key)

print(np.array(data['meta']['Description']), np.array(data['meta']['Detector']))


# Import packages
import numpy as np
import h5py
# Assign filename: file
file = 'LIGO_data.hdf5'
# Load file: data
data = h5py.File(file, 'r')
# Print the datatype of the loaded file
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)


# Get the HDF5 group: group
group = data['strain']
# Check out keys of group
for key in group.keys():
    print(key)
# Set variable equal to time series data: strain
strain = np.array(data['strain']['Strain'])
# Set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()


# MATLAB and .mat files
scipy.io.loadmat() - read .mat files
scipy.io.savemat() - write .mat files

import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))

"""keys = matlab variable names
values = objects asssigned to variables"""



# Import package
import scipy.io
# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')
# Print the datatype type of mat
print(type(mat))
# Print the keys of the MATLAB dictionary
print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))
# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()


# Relational data bases
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Nothwind.sqlite')
table_names = engine.table_names()
print(table_names)


# Import necessary module
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Save the table names to a list: table_names
table_names = engine.table_names()
# Print the table names to the shell
print(table_names)


"""
WORKFLOW OF SQL QUERYING
- import packages and functions
- create the database engine
- connect to the engine
- Query the database
- Save query results to a DataFrame
- Close the connection
"""

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Orders")
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

print(df.head())

# You can use the context manager
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect as con:
	rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
	df = pd.DataFrame(rs.fetchmany(size=5))
	df.columns = rs.keys()


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()
# Print the length of the DataFrame df
print(len(df))
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee where EmployeeId >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())
    # Set the DataFrame's column names
    df.columns = rs.keys()
# Print head of DataFrame
print(df.head())



#pandas can do this in one line.
df = pd.read_sql_query("SELECT * FROM Orders", engine)



# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Album", engine)
# Print head of DataFrame
print(df.head())
# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
# Confirm that both methods yield the same result
print(df.equals(df1))


# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Employee WHERE EmployeeId >= 6 Order by Birthdate",engine)
# Print head of DataFrame
print(df.head())


# INNER JOIN in Pandas
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
df = pd.read_sql_query("SELECT OrderID, CompanyName From Orders \
	INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID", engine)
print(df.head())


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print head of DataFrame df
print(df.head())


# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000",engine)
# Print head of DataFrame
print(df.head())


# urllib
urlopen() # Accepts URLs instead of file names

from urllib.request import urlretrieve
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve(url, 'winequality-white.csv') # write this url to a file


#Reading csvs from the web or url using urlretrieve

# Import package
from urllib.request import urlretrieve
# Import pandas
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
urlretrieve(url, 'winequality-red.csv')
# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())


#Using pandas and a url to a csv from the web and save it as a data frame without saving it locally.

# Import packages
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')
# Print the head of the DataFrame
print(df.head())
# Plot first column of df
df.iloc[:, 0].hist()
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


# Importing non-flat files from the web

# Importing Excel files from the web 
# Import package
import pandas as pd
# Assign url of file: url
url = 'https://assets.datacamp.com/course/importing_data_into_r/latitude.xls'
# Read in all sheets of Excel file: xls
xls = pd.read_excel(url, sheet_name = None)
# Print the sheetnames to the shell
print(xls.keys())
# Print the head of the first sheet (using its name, NOT its index)
print(xls['1700'].head())


# HTTP Requests to import files from the web
from urllib.request import urlopen, Request
url = "http://www.wikipedia.org/"
request = Request(url)
response = urlopen(request) 
html = response.read()
response.close()

# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request: request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Print the datatype of response
print(type(response))
# Be polite and close the response!
response.close()


# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Extract the response: html
html = response.read()
# Print the html
print(html)
# Be polite and close the response!
response.close()




import requests
url = "http://www.wikipedia.org/"
r = requests.get(url)
text = r.text


# Import package
import requests
# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"
# Packages the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response: text
text = r.text
# Print the html
print(text)


# Beautiful Soup
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BeautifulSoup'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
print(soup.prettify())

soup.title
soup.get_text()
soup.find_all()

for link in soup.find_all('a'):
	print(link.get('href'))


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()
# Print the response
print(pretty_soup)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Get the title of Guido's webpage: guido_title
guido_title = soup.title
# Print the title of Guido's webpage to the shell
print(guido_title)
# Get Guido's text: guido_text
guido_text = soup.get_text()
# Print Guido's text to the shell
print(guido_text)


# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Print the title of Guido's webpage
print(soup.title)
# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


# APIs and JSONs
# JSON are dictionaries
import json
with open('snakes.json', 'r') as json_file:
	json_data = json.load(json_file)
type(json_data) # dictionary
for key, value in json_data.items():
	print(key + ':', value)


# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


import requests
url = 'http://www.omdbapi.com/?t=hackers'
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
	print(key + ':', value)



# Import requests package
import requests
# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Print the text of the response
print(r.text)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# Import package
import requests
# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


# Twitter
tweets.py
import tweepy, json
access_token = "..."
access_token_secret = '...'
consumer_key = '...'
consumer_secret = '...'

stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
stream.filter(track=['apples', 'oranges'])


# Store credentials in relevant variables
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
# Create your Stream object with credentials
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
# Filter your Stream variable
stream.filter(['clinton', 'trump', 'sanders', 'cruz'])


# Twitter API
# Import package
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Print the keys of the first tweet dict
print(tweets_data[0].keys())
# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
# Print head of DataFrame
print(df.head())

import re
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False


# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot the bar chart
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()








