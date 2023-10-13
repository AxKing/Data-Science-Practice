# Fundamentals of Big Data
# With PySpark

### Chapter 1
### Section 1
### Fundamentals of Big Data

# Volume - The ammount of data
# Variety - The different sources of data and the different types
# Velocity- The speed at which data is generated 

# Clustered computing - collection of resources of multiple machines
# Parallel computing - simultaneous computation
# Distributed computing - Collection of nodes (networked computers) that run in parallel
# Batch processing - Breaking the job into small pieces and running them on individual machines
# Real-Time processing - Immediate processing of data

# Hadoop/MapReduce - Scalable and fault tolerabt framework written in Java
	# Open source
	# Batch Processing
# Aparche Spark - General Purpose and lightning fast cluster computing system
	# Open Source
	# both batch and real time data processing

# Spark
	# Distributed cluster computing framework
	# efficient in-memory computations for large data sets
	# Lightning fast data processing framework
	# Provides support for Java, Scala, Python, R and SQL
# Spark Components
	# SPark SQL
	# MLib Machine Learning - 
	# GraphX - algorithms for graphs and tools
	# Spark Streaming - real time data
	# RDD API Apache Spark Core
# local mode - single machine like a laptop
# cluster mode - good for production
# workflow local -> clusters


### Chapter 1
### Section 1
### Exercies
# None


### Chapter 1
### Section 2
### PySpark: Spark with Python

# Interactive environment for running spark jobs
# Helpful for fast interactive prototyping
# 3 different shells
# PySpark shell support connecting to a cluster
# SparkContext is an entry point into the world of Spark
# An entry point is like a key to the house

sc.version # what version of spark
sc.pythonVer # what version of python
sc.master # what is the connecting URL?

# Loading data into pyspark
parallelize()
rdd = sc.parallelize([1,2,3,4,5])

rdd2 = sc.textFile('test.txt')


### Chapter 1
### Section 2
### Exercies

# Print the version of SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)

# Print the Python version of SparkContext
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)

# Print the master of SparkContext
print("The master of Spark Context in the PySpark shell is", sc.master)


# Create a Python list of numbers from 1 to 100 
numb = range(1, 101)

# Load the list into PySpark  
spark_data = sc.parallelize(numb)


# Load a local file into PySpark shell
lines = sc.textFile(file_path)


### Chapter 1
### Section 3
### Use of Lambda function in Python filter()

lambda arguments: expression
double = lambda x: x * 2
print(double(3))

def cube(x):
	return x ** 3
g = lambda x: x ** 3

print(g(10))
print(cube(10))

map() - function takes a function and a list 
and returns a new list which contains items returned by that function for each item

# Map Syntax
map(function, list)
EX:
items = [1,2,3,4]
list(map(lambda x: x+2, items))


filter() - function takes a function and a list and returns a 
new list for which the function evaluates as true

filter(function, list)
EX:
items = [1,2,3,4]
list(filter(lambda x: (x%2 != 0), items))


### Chapter 1
### Section 3
### Exercies

# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list
squared_list_lambda = list(map(lambda x: x ** 2, my_list))

# Print the result of the map function
print("The squared numbers are", squared_list_lambda)



# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)




#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

### Chapter 2
### Section 1
### Abstracting Dada with RDDs

# RDD - resilient Distributed Datasets
# Resilient - withstands failures
# Distributed - spanning across multiple nodes
# Datasets - Collection of partitioned data eg arrays, tables, tuples, etc

# How to create RDDs?
Parallelizing an existing collection of objects
External datasets
	Files in HDFS
	Objects in Amazon S3 bucket
	Lines in a text file
From Existing RDDs

numRDD = sc.parallelize([1,2,3,4])
helloRDD = sc.parallelize('Hello World')

type(HelloRDD)

textFile()
fileRDD = sc.textFile('README.md')
type(fileRDD)

- A partition is a logical division of a large distributed data set
# paralellize()
numRDD = sc.parallelize(range(10), minPartitions = 6)
# textFile()
fileRDD = sc.textFile('README.md', minPartitions = 6)

getNumPartitions()

### Chapter 2
### Section 1
### Exercies

# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])
# Print out the type of the created object
print("The type of RDD is", type(RDD))


 # Print the file_path
print("The file_path is", file_path)
# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)
# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))


# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())
# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)
# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

### Chapter 2
### Section 2
### RDD Operations in PySpark

Transformations and Actions

# Transformations create new RDDs
Transformations follow Lazy evaluation
map, filer, flatMap and union

map() tranformation applies a function to all elements in the RDD
RDD = sc.parallelize([1,2,3,4])
RDD_map = RDD.map(lambda x: x*x)

filter() Thefilter transformation returns a new RDD with only the elements that pass the condition
RDD = sc.parallelize([1,2,3,4])
RDD_filter = RDD.filter(lambda x: x > 2)

flatMap() transformation returns multile values for each element in the original RDD
RDD = sc.parallelize(['hello world', 'how are you'])
RDD_flatmap = RDD.flatMap(lambda x: x.split(" "))

union() Takes two existing RDDs and combines them into one.
inputRDD = sc.textFile('logs.txt')
errorRDD = inputRDD.filter(lambda x: "error" in x.split())
warningsRDD = inputRDD.filter(lambda x: "warnings" in x.spit())
combineRDD = errorRDD.union(warningsRDD)

# Actions Perform computation on the RDDs
Operations return a value after running a computation on the RDD

collect() # returns all elements of the dataset as an array
RDD_map.collect()

take(N) # returns an array with the first N elements of the dataset
RDD_map.take(2)

first() # returns the first element of the RDD
RDD_map.first()

count() # returns the total number of rows in the RDD
RDD_flatmap.count()


### Chapter 2
### Section 2
### Exercies

# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x * x * x)
# Collect the results
numbers_all = cubedRDD.collect()
# Print the numbers from numbers_all
for numb in numbers_all:
	print(numb)


# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in line.split())
# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())
# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4): 
  print(line)




### Chapter 2
### Section 3
### Working with key/value Pairs RDDs in PySpark

Pair RDD - each row is a key and maps to one or more values
Pair RDD is a special data structure to work with this kind of dataset
key identifier
value data

Creating pair RDDs
	From a list of key-value tuples
	From a regular RDD
my_tuple = [('Sam', 23), ('Mary',34), ('Peter',25)]
pairRDD_tuple = sc.parallelize(my_tuple)

my_list = ['Sam 23', 'Mary 34', 'Peter 25']
regularRDD = sc.parallelize(my_list)
pairRDD = regularRDD.map(lambda s: (s.split(' ')[0], s.plit(' ')[1]))


Transformation on pair RDDs
reduceByKey() Combine values with the same key
groupByKey() Group Values with the same key
sortByKey() Return an RDD sorted by the key
join() Join two pair RDDs based on their key

reduceByKey() # Transformation combines values with the same key
# it runs parallel operations for each key in the dataset
# it is a transformation and not an action
regularRDD = sc.parallelize([('Messi', 23),('Ronaldo', 34),('Neymar', 22),('Messi', 24)])
pairRDD_reducebykey = regularRDD.reduceByKey(lambda x,y: x + y)
pairRDD_reducebykey.collect()
#Returns: [('Neymar', 22), ('Ronaldo', 34), ('Messi', 47)]


sortByKey() # Operation oders pair RDD by Key
pairRDD_reducebykey_rev = parRDD_reducebykey.map(lambda x: (x[1], x[0]))
pairRDD_reducebykey_rev.sortByKey(ascending=False).collect()
[(47, 'Messi'), (34, 'Ronaldo'), (22, 'Neymar')]

groupByKey() # groups all the values with the same key in the pair RDD
airports = [('US', 'JFK'), ('UK', 'LHR'), ('FR', 'CDG'), ('US', 'SFO')]
regularRDD = sc.parallelize(airports)
pairRDD_group = regularRDD.groupByKey().collect()
for cont, air in pairRDD_group:
	print(cont, list(air))
FR ['CDG']
US ['JFK', 'SFO']
UK ['LHR']

join() # transformation joins the two pair RDDs based on their key
RDD1 = sc.parallelize([('Messi', 34),('Ronaldo', 32),('Neymar', 24)])
RDD2 = sc.parallelize([('Ronaldo', 80),('Neymar', 120),('Messi', 100)])
RDD1.join(RDD2).collect()
[('Neymar', (24,120)), ('Ronaldo',(32,80)), ('Messi',(34,100))]



### Chapter 2
### Section 3
### Exercies

# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1,2),(3,4),(3,6),(4,5)])
# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: x + y)
# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

# Sort the reduced RDD with the key by descending order
Rdd_Reduced_Sort = Rdd_Reduced.sortByKey(ascending=False)
# Iterate over the result and retrieve all the elements of the RDD
for num in Rdd_Reduced_Sort.collect():
  print("Key {} has {} Counts".format(num[0], num[1]))


### Chapter 2
### Section 4
### Advanced RDD Actions

reduce() reduce(func) action is used for aggregating the elements of a regular RDD
# The function should be commutative and associative
EX: +
x = [1,3,4,6]
RDD = sc.parallelize(x)
RDD.reduce(lambda x,y: x+y)


saveAsTextFile() # Turns RDD into a text file inside a directory with each partition as a separate file.
RDD.saveAsTextFile("tempFile")

coalesce() # method can be used to save RDD as a sigle text file
RDD.coalesce(1).saveAsTextFile("tempFile")

Pair RDD actions
countByKey() #only available for type (K,V)
countByKey() # action counts the number of elements for each key
rdd = sc.paralellize([('a', 1), ('b',2), ('a',1)])
for kee, val in rdd.countByKey().items():
	print(kee, val)

collectAsMap() # return the Key-Value pairs in the RDD as a dictionary
EX
sc.parallelize([(1,2), (3,4)]).collectAsMap()



### Chapter 2
### Section 4
### Exercies

# Count the unique keys
total = Rdd.countByKey()
# What is the type of total?
print("The type of total is", type(total))
# Iterate over the total and print the output
for k, v in total.items(): 
  print("key", k, "has", v, "counts")


# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)
# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())
# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())


# Convert the words in lower case and remove stop words from the stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)
# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))
# Count of the number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)


# Display the first 10 words and their frequencies from the input RDD
for word in resultRDD.take(10):
	print(word)
# Swap the keys and values from the input RDD
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[0]))
# Sort the keys in descending order
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)
# Show the top 10 most frequent words and their frequencies from the sorted RDD
for word in resultRDD_swap_sort.take(10):
	print("{},{}". format(word[1], word[0]))

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

### Chapter 3
### Section 1
### Introduction to PySpark DataFrames SQL

# schemas and dataframes
# creating a dataframe requires an RDD and a schema
createDataFrame(RDD, schema)

iphones_RDD = sc.parallelize([
	("XS", 2018, 5.56, 2.79, 6.24),
	("XR", 2018, 5.94, 2.98, 6.84),
	("X10", 2017, 5.56, 2.79, 6.13),
	("8Plus", 2017, 6.23, 3.07, 7.12)
])
names = ['Model', 'Year', 'Height', 'Width', 'Weight']
iphones_df = spark.createDataFrame(iphones_RDD, schema = names)
type(iphones_df)


# Create one from a CSV
df_csv = spark.read.csv("people.csv", header=True, inferSchema=True)
df_json = spark.read.json('people.json', header=True, inferSchema=True)
df_txt = spark.read.txt("people.txt", header=True, inferSchema = True)




### Chapter 3
### Section 1
### Exercies

# Create an RDD from the list
rdd = sc.parallelize(sample_list)
# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema=['Name', 'Age'])
# Check the type of names_df
print("The type of names_df is", type(names_df))


# Create an DataFrame from file_path
people_df = spark.read.csv(file_path, header=True, inferSchema=True)
# Check the type of people_df
print("The type of people_df is", type(people_df))


### Chapter 3
### Section 2
### Operating on DataFrames in Pyspark

# Transformations and Actions
# Tranformations:
select(), filter(), groupby(), orderby(), dropDuplicates(), withColumnRenamed()
#Actions
printSchema(), head(), show(), count(), columns(), describe()

select() # select data by inputting a column
df_id_age = test.select('Age')
df_id_age.show(3)

filter() # transformation filters out the rows based on a condition
new_df_age21 = new_df.filter(new_df.Age > 21)
new_df_age21.show(3)

groupby() # operation can be used to group a variable
test_df_age_group = test_df.groupby('Age')
test_df_age_group.count().show(3)

orderby() # operation sorts the DataFrame based on one or more columns
test_df_age_group.count().orderBy('Age').show()

dropDuplicates() # removes the duplicate rows of a DataFrame
test_df_no_dup = test_df.select('User_ID', 'Gender', 'Age').dropDuplicates()
test_df_no_dup.count()

withColumnRenamed() # returns a new dataframe with a column renamed
test_df_sex = test_df.withColumnRenamed('Gender', 'Sex')
test_df_sex.show(3)

printSchema() # operation prints the types of columns in the DataFrame
test_df.printSchema()

columns #operator prints the columns of a DataFrame
test_df.columns 

describe() #operation compute summary statistics of numerical columns
test_df.describe().show()


### Chapter 3
### Section 2
### Exercies

# Print the first 10 observations 
people_df.show(10)
# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))
# Count the number of columns and their names
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))


# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')
# Print the first 10 observations from people_df_sub
people_df_sub.show(10)
# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()
# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))


# Filter people_df to select females 
people_df_female = people_df.filter(people_df.sex == "female")
# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")
# Count the number of rows 
print("There are {} rows in the people_df_female DataFrame and {} rows in the people_df_male DataFrame".format(people_df_female.count(), people_df_male.count()))


### Chapter 3
### Section 3
### Interacting with DataFrames using PySpark SQL

sql() # takes a sql statement as an argument and returns the result as a DataFrame

df.createOrReplaceTempView('table1')

df2 = spark.sql("SELECT field1, field2 FROM table1")
df2.collect()


test_df.createOrReplaceTempView('test_table')
query = '''SELECT Product_ID FROM test_table '''
test_product_df = spark.sql(query)
test_product_df.show(5)

test_df.createOrReplaceTempView('test_table')
query = """SELECT Age, max(Purchase) FROM test_table GROUP BY Age """
spark.sql(query).show(5)

#Filtering columns using SQL Queries
test_df = createOrReplaceTempView('test_table')
query = '''Age Purchase, Gender FROM test_table WHERE Purchase > 20000 AND Gender =="F" '''
spark.sql(query).show(5)



### Chapter 3
### Section 3
### Exercies

# Create a temporary table "people"
people_df.createOrReplaceTempView("people")
# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''
# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)
# Print the top 10 names of the people
people_df_names.show(10)


# Filter the people table to select female sex 
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')
# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * FROM people WHERE sex=="male"')
# Count the number of rows in both DataFrames
print("There are {} rows in the people_female_df and {} rows in the people_male_df DataFrames".format(people_female_df.count(), people_male_df.count()))


### Chapter 3
### Section 4
### Data Visualization in PySpark using DataFrames

pyspark_dist_explore library
toPandas() method
HandySpark library

Pyspark_dist_explore
	hist()
	distplot()
	pandas_histogram()

hist()
test_df = spark.read.csv('test.csv', header = True, inferSchema = True)
test_df_age = test_df.select('Age')
hist(test_df_age, bins = 20, color = 'red')

toPandas()
test_df = spark.read.csv('test.csv', header = True, inferSchema = True)
test_df_sample_pandas = test_df.toPandas()
test_df_sample_pandas.hist('Age')

HandySpark method of visualization
test_df = spark.read.csv('test.csv', header = True, inferSchema = True)
hdf = test_df.toHandy()
hdf.cols['Age'].hist()


### Chapter 3
### Section 4
### Exercies

# Check the column names of names_df
print("The column names of names_df are", names_df.columns)
# Convert to Pandas DataFrame  
df_pandas = names_df.toPandas()
# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='Name', y='Age', colormap='winter_r')
plt.show()



# Load the Dataframe
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)
# Check the schema of columns
fifa_df.printSchema()
# Show the first 10 observations
fifa_df.show(10)
# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))


# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')
# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''
# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)
# Generate basic statistics
fifa_df_germany_age.describe().show()


# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()
# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

### Chapter 4
### Section 1
### Overview of PySpark MLlib

# MLlib is a component of Apache Spark for machine learning
Collaborative # - Produces Recomendations
Classification # - Identifying to which of a set of categories a new observation
Clustering # - Groups data based on similar characteristics

from pyspark.mllib.recommendation import ALS

from pyspark.mllib.classification import LogisticRegressionWithLBFGS

from pyspark.mllib.clustering import KMeans



### Chapter 4
### Section 1
### Exercies

# Import the library for ALS
from pyspark.mllib.recommendation import ALS

# Import the library for Logistic Regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

# Import the library for Kmeans
from pyspark.mllib.clustering import KMeans





### Chapter 4
### Section 2
### Introduction to Collaborative Filtering

# Collaborative Filtering - method is finding users that share common interests
user-user 
item-item 

# The rating class in pyspark.mllib.recommendation submodule
# The Rating class is a wrapper around the tuple, (user, product, rating)

from pyspark.mllib.recommendation import Rating
r = Rating(user = 1, product = 2, rating = 5.0)
(r[0], r[1], r[2])

randomSplit()
data= sc.parallelize([1,2,3,4,5,6,7,8,9,10])
training, test=data.randomSplit([0.6,0.4])
training.collect()
test.collect()

# Alternating Least Squares 
spark.mllib
REQUIRES 
ALS.train(ratings, rank, iterations)
# rank is the number of products
EX
r1 = Rating(1, 1, 1.0)
r2 = Rating(1, 2, 2.0)
r3 = Rating(2, 1, 2.0)
ratings = sc.parallelize([r1,r2,r3])
ratings.collect()
model = ALS.train(ratings, rank=10, iterations=10)

predictAll()
unrated_RDD = sc.parallelize([(1,2), (1,1)])
predictions = model.predictALL(unrated_RDD)
predictions.collect()


#Evaluation MSE
Mean Square Error
rates = ratings.map(lambda x:((x[0], x[1]), x[2] ))
preds = predictions.map(lambda x:((x[0], x[1]), x[2] ))
preds.collect()

rates_preds = rates.join(preds)
rates_preds.collect()

MSE = rates_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()

### Chapter 4
### Section 2
### Exercies 

# Load the data into RDD
data = sc.textFile(file_path)
# Split the RDD 
ratings = data.map(lambda l: l.split(','))
# Transform the ratings RDD 
ratings_final = ratings.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))
# Split the data into training and test
training_data, test_data = ratings_final.randomSplit([0.8, 0.2])


# Create the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)
# Drop the ratings column 
testdata_no_rating = test_data.map(lambda p: (p[0], p[1]))
# Predict the model  
predictions = model.predictAll(testdata_no_rating)
# Return the first 2 rows of the RDD
predictions.take(2)


# Prepare ratings data
rates = ratings_final.map(lambda r: ((r[0], r[1]), r[2]))
# Prepare predictions data
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))
# Join the ratings data with predictions data
rates_and_preds = rates.join(preds)
# Calculate and print MSE
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))


### Chapter 4
### Section 3
### Classification

# Classification
# Logistic Regression
# Yes or No
# Vector and LabelledPoint
Dense vector - store all their entries in an array of floating point numbers
Sparse vectors - store only the nonzero values and their indicies

denseVec = Vectors.dense([1.0,2.0,3.0])
sparseVec = Vectors.sparse(4, {1:1.0, 3:5.5})

LabelledPoint is a wrapper for input features and predicted value
positive = LabeledPoint(1.0, [1.0, 0.0, 3.0])
negative = LabeledPoint(0.0, [2.0, 1.1, 1.0])

HashingTF() algorithm is used to map feature value to indicies in the feature vector
EX
from pyspark.mllib.feature import HashingTF
sentence = 'helo hello world'
words = sentence.split()
tf = Hashing(10000) # (tf = term filter)
tf.transform(words)

SparseVector(10000, {3065: 1.0, 6861: 2.0})

# Logistic Regression using LogisticRegressionWithLBFGS
data = [
	LabeledPoint(0.0, [0.0,1.0]),
	LabeledPoint(1.0, [1.0,0.0])
	]
RDD = sc.parallelize(data)

lrm = LogisticRegressionWithLBFGS.train(RDD)
lrm.predict([1.0, 0.0])
lrm.predict([0.0, 1.0])



### Chapter 4
### Section 3
### Exercies

# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)
# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))
# Print the first element in the split RDD
print("The first element in spam_words is", spam_words.first())
print("The first element in non_spam_words is", non_spam_words.first())


# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)
# Map each word to one feature
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)
# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))
# Combine the two datasets
samples = spam_samples.join(non_spam_samples)


# Split the data into training and testing
train_samples,test_samples = samples.randomSplit([0.8, 0.2])
# Train the model
model = LogisticRegressionWithLBFGS.train(train_samples)
# Create a prediction label from the test data
predictions = model.predict(test_samples.map(lambda x: x.features))
# Combine original labels with the predicted labels
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)
# Check the accuracy of the model on the test data
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))


### Chapter 4
### Section 4
### Clustering

RDD = sc.textFile('WineData.csv').map(lambda x: x.split(',')).map(lambda x: [float(x[0])])
RDD.take(5)

KMeans.train()
EX
from pyspark.mllib.clustering import KMeans
model = KMeans.train(RDD, k=2, maxIterations = 10)
model.clusterCenters

#Evaluate K-Means Model
from math import sqrt
def error(point):
	center = model.centers[model.predict(point)]
	return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = RDD.map(lambda point: error(point)).reduct(lambda x,y: x+y)
print("Within Set Sum of Squared Error = " +str(WSSSE))

# Visualizing K-Means
#Convert the RDD to a Pandas Data Frame
wine_data_df = spark.createDataFrame(RDD, schema=['col1', 'col2'])
wine_data_df_pandas = wine_data_df.toPandas()
# Conveert the Centers
cluster_centers_pandas = pd.DataFrame(model.clusterCenters, columns=['col1', 'col2'])
cluster_centers_pandas.head()

plt.scatter(wine_data_df_pandas['col1'], wine_data_df_pandas['col2']);
plt.scatter(cluster_centers_pandas['col1'], cluster_centers_pandas['col2'], color='red', marker='x')

### Chapter 4
### Section 4
### Exercies

# Load the dataset into an RDD
clusterRDD = sc.textFile(file_path)
# Split the RDD based on tab
rdd_split = clusterRDD.map(lambda x: x.split("\t"))
# Transform the split RDD by creating a list of integers
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])
# Count the number of rows in RDD 
print("There are {} rows in the rdd_split_int dataset".format(rdd_split_int.count()))


# Train the model with clusters from 13 to 16 and compute WSSSE
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k
model = KMeans.train(rdd_split_int, k=15, seed=1)
# Get cluster centers
cluster_centers = model.clusterCenters
# Convert rdd_split_int RDD into Spark DataFrame and then to Pandas DataFrame
rdd_split_int_df_pandas = spark.createDataFrame(rdd_split_int, schema=["col1", "col2"]).toPandas()
# Convert cluster_centers to a pandas DataFrame
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])
# Create an overlaid scatter plot of clusters and centroids
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()
