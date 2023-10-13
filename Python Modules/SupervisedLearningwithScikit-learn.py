# supervised learning with scikit-learn

"""
There are two types of supervised learning.
Classification
and 
Regression

Features- predictor variable- independent variable
Target Variable - dependent variable - response variable


SUPERVISED LEARNING REQUIREMENTS
* NO MISSING VALUES
* DATA IN NUMERIC FORMAT
* DATA STORED IN PANDAS Dataframe or NumPy array

Perform Exploratory Data Analysis (EDA) first

"""

# Chapter 1, Section 1
# scikit-learn syntax
from sklearn.module import Model #Import a model like k-nearest-neighbor
model = Model() # we instantiate the model
model.fit(X,y) # We fit the model to X- an array of features, y- an array of target variables
predictions = model.predict(X_new) # We pass in new observations X_new and receive an array of predictions
print(predictions) # 0 is no, 1 is yes


# Chapter 1, Section 2
# The Classification Challenge
"""
Classifying lables of unseen data
1. Build a model
2. Model learns from the labeled data we pass it
3. Pass unlabeled data to the model as input
4. Model predicts the lables of the unseen data

K-Nearest Neighbors KNN
Predict the label of a data point by
	* Looking at the K cloest labeled data points
	* Taking a majority vote
	* 
"""
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[['total_day_charge', 'total_eve_chage']].values # required to be in an array
y = churn_df['churn'].values # required to be in a single column
print(X.shape, Y.shape) # these must have the same number of entries.

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)

X_new = np.array([[56.8, 17.5],
				[24.4, 24.1],
				[50.1, 10.9]])
print(X_new.shape)
predictions = knn.predict(X_new)

print('Predictions: {}'.format(predictions))


# EXERCISES
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values
# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)
# Fit the classifier to the data
knn.fit(X, y)
# Predict the labels for the X_new
y_pred = knn.predict(X_new)
# Print the predictions for X_new
print("Predictions: {}".format(y_pred)) 


# Chapter 1, section 3
# Measuring Model performance
"""
In classification, accuracy is a commonly used metric
Accuracy: correct predictions / total observations

Could compute accuracy on the data used to fit the classifier.

Split the data into a training set, and a test set.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, \
	random_state = 21, stratify = y)
# X 
# y
# test_size 30% of the data for the test
# random_state This is a specific seed of random number so we can duplicate the results.
# stratify If 10% of the observations are yes, we want 10% of the training data to be yes.
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test)) #The score method gives a percentage of acruaracy.

# Larger K = less complex => Underfitting
# Smaller K = more complex => Overfitting

# Model Complexity and oer/underfitting
train_accuracies = {}
test_accuracies = {}
neighbors = np.arrange(1,26)
for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors=neighbor)
	knn.fit(X_train, y_train)
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()



####
#EXERCISE
# Import the module
from sklearn.model_selection import train_test_split
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Print the accuracy
print(knn.score(X_test, y_test))

# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}
for neighbor in neighbors:
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)  
	# Fit the model
	knn.fit(X_train, y_train)
  	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)


# Add a title
plt.title("KNN: Varying Number of Neighbors")
#Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
# Display the plot
plt.show() 







# Chapter 2 Introduction to Regression
# Section 1 

import pandas as pd
diabetes_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head())

# we need to have arrays without our target variables
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df['glucose'].values

# We're going to try and predict using one variable
X_bmi = X[:,3]
# For scikitlearn, we need a 2-d array
X_bmi = X_bmi.reshape(-1,1)

import matplotlib.pyplot as plot
plt.scatter(X_bmi, y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# Now we can plot the data and the best fit line
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()


##### Chapter 2, Section 1
##### Exercises
import numpy as np

# Create X from the radio column's values
X = sales_df['radio'].values
# Create y from the sales column's values
y = sales_df['sales'].values
# Reshape X
X = X.reshape(-1,1)
# Check the shape of the features and targets
print(X.shape, y.shape)

# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Create the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X, y)
# Make predictions
predictions = reg.predict(X)
print(predictions[:5])

# Import matplotlib.pyplot
import matplotlib.pyplot as plt
# Create scatter plot
plt.scatter(X, y, color="blue")
# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")
# Display the plot
plt.show()


##### Chapter 2, Section 2
##### The basics of linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

reg_all.score(X_test, y_test)

# To calculate the Root Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False)


#### CH2, Sec 2 Exercises
# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Instantiate the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X_train, y_train)
# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Import mean_squared_error
from sklearn.metrics import mean_squared_error
# Compute R-squared
r_squared = reg.score(X_test, y_test)
# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


##### Ch2, Ch3
##### Cross-Validation

from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle = True, random_state=42)
reg = LinearRegression()
cv_results = print(cross_val_score(reg, X, y, cv=kf))

# Print the error
np.mean(cv_results)
np.std(cv_results)
np.quantile(cv_results,[0.025, 0.975])



# Exercises
# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold
#Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)
reg = LinearRegression()
# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)
# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_results))
# Print the standard deviation
print(np.std(cv_results))
# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))


##### Ch2 Sec 3
##### Regularized Regression

# Ridge Method uses Ordinary Least Squares loss function
# Ridge method punishes large positive and large negative coefficients

# We select an alpha value
# Similarly we select a K value

# Hyperparameter: Variable used to optimize model parameters
# Alpha controls complexity
# A = 0 = OLS (Can lead to overfitting)
# Very High A can lead to underfitting

from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
	ridge = Ridge(alpha=alpha)
	ridge.fit(X_train, y_train)
	y_pred = ridge.predict(X_test)
	scores.append(ridge.score(x_test, y_test))
print(scores)


# LASSO Regression
# Lasso regression shrinks coefficients of less important features to zero
from sklear.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
	lasso = Lasso(alpha=alpha)
	lasso.fit(X_train, y_train)
	lasso_pred = lasso.predict(X_test)
	scores.append(lasso.score(X_test, y_test))
print(scores)


# Lasso can select important features of a dataset
from sklearn.linear_model import Lasso
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df['glucose'].values
names = diabetes_df.drop("glucose", axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.xticks(rotation=45)
plt.show()


##### Exercises
# Import Ridge
from sklearn.linear_model import Ridge
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:  
  # Create a Ridge regression model
  ridge = Ridge(alpha=alpha)  
  # Fit the data
  ridge.fit(X_train, y_train)
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)



# Import Lasso
from sklearn.linear_model import Lasso
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)
# Fit the model to the data
lasso.fit(X, y)
# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()







##### Chapter 3
##### Section 1
##### How good is your model?

"""
Confusion Matrix is a 2 x2 matrix
It gives 4 values

Accuracy is the sum of the correct predictions divided by the total number of predictions
in the matrix
t_p + t_n / t_p + t_n + f_p + t_n


Precision (High Precision means a higher rate of true positives)
minimize false negatives
t_p / t_p + f_p

Recall or Sensitivity
t_p/t_p + f_n
High Recall = lower negative rate
High Recall = Predicted most frudulent transactions correctly

F1 score takes into account precision and recall
"""

from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors = 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)

[[116, 35]
[  46, 34 ]]
116 true negatives 
35 false positives
34 true positives
46 false negatives


classification_report(y_test, y_pred)
# Gives precision, recall, f1-score, support, accuracy, macro avg, weighted avg


##### Chapter 3
##### Section 2
##### Logistic Regression and the ROC Curve

"""
Logiostic Regression gives a probability that an output belongs to a certain class
Classification Model
"""
from sklearn.linear_model import LogisticRegression # import the model
logreg = LogisticRegression() # instantiate the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) # Split our data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# You can predict probabilities by calling the predict_proba method
y_pred_probs = logreg.predict_proba(X_test)[:,1]

# receiver operating characteristic curve or ROC curve is a graphical plot that
# illustrates the diagnostic ability of a binary classifier

# To plot the ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
# fpr is false positive rate
# trp true positive rate
plt.plot([0,1], [1,0], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rage')
plt.title('Logistic Regression ROC Curve')
plt.show()


# ROC AUC
# Area Under the ROC Curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_probs)


##### Exercises
#Import LogisticRegression
from sklearn.linear_model import LogisticRegression
# Instantiate the model
logreg = LogisticRegression()
# Fit the model
logreg.fit(X_train, y_train)
# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[:10])


# Import roc_curve
from sklearn.metrics import roc_curve
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0, 1], [0, 1], 'k--')
# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()


# Import roc_auc_score
from sklearn.metrics import roc_auc_score
# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))
# Calculate the classification report
print(classification_report(y_test, y_pred))


##### Chapter 3
##### Section 3
##### Hyperparameter Tuning

"""
Hyperparameters: Parameters we specify before fitting the model
We have seen alpha, and n_neighbors

Hyperparameter Tuning
1. Try lots of different hyperparameter values
2. Fit all of them separately
3. See how well they perform
4. Choose the best performing values

* It is essential to use cross-validation to avoid overfitting to the test set
* We can still split the data and perform cross-validation on the training set

Grid Search
"""
# Performing a Grid Search on a regression model using our sales data set
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arrange(0.0001, 1, 10),
							'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cs.best_score_)


# Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha':np.arrage(0.0001, 1, 10),
							'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

test_score = ridge_cv.score(X_test, y_test)

##### Exercises
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
#Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}
# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)
# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))


#Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}
# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)
# Fit the data to the model
logreg_cv.fit(X_train, y_train)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))


##### Chapter 4
##### Section 1
##### Preprocessing Data

# Dealing with categorical features
# Convert to binary features called dummy variables
sklearn has OneHotEncoder()
Pandas: get_dummies()

print(music.info)

import pandas as pd
music_df = pd.read_csv('music.csv')
# get_dummies breaks a column into dummy variables
music_dummies = pd.get_dummies(music_df['genre'], drop_first = True)
# We need to concat our new dummies with our old data set
music_dummies = pd.concat([music_df, music_dummies], axis=1)
# Now we drop the genre column from our data frame
music_dummies = music_dummies.drop('genre', axis=1)

# If the data frame only has one categorical column, you can skip the combination
music_dummies = pd.get_dummies(music_df, drop_first=True)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
X = music_dummies.drop('popularity', axis=1).values
y = music_dummies['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoreing='neg_mean_squared_error')
print(np.sqrt(-linreg_cv))


##### Exercises

# Create music_dummies
music_dummies = pd.get_dummies(music_df, drop_first=True)
# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))

# Create X and y
X = music_dummies.drop('popularity', axis=1).values
y = music_dummies['popularity'].values
#Instantiate a ridge model
ridge = Ridge(alpha=0.2)
#Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")
#Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))



##### Chapter 4
#####	Section 2
##### Handling Missing Data

music_df.isna().sum().sort_values()

# Drop missing data
music_df = music_df.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
# check again
music_df.isna().sum().sort_values()


# Impute Missing Values
mean
median
mode

# For imputation
from sklearn.impute import SimpleImputer
X_cat = music_dff['genre'].values.reshape(-1,1)
X_num = music_df.drop(['genre', 'popularity'], axis=1).values
y = music_df['popularity'].values
# Categorical values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=12)
# Numerical
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size =0.2, random_state = 12)

# imputing missing categorical data
imp_cat = SimpleImputer(stragety='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)
# imputing numerical data
imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.fit_transform(X_test_num)
# Then combine the data
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis = 1)

# Imputing within a pipeline 
from sklearn.pipeline import Pipeline
music_df = music_df.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values

steps = [('imputation', SimpleImputer()), ("logistic_regression", LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)



#Exercises
# Print missing values for each column
print(music_df.isna().sum().sort_values())
# Remove values where less than 5% are missing
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)
print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))


steps = [("imputer", imp_mean),
        ("knn", knn)]
# Create the pipeline
pipeline = Pipeline(steps)
# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
# Make predictions on the test set
y_pred = pipeline.predict(X_test)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))





##### Chapter 4
##### Section 3
##### Center and Scaling

music_df['duration_ms', 'loudness'].describe()

# Many models use some for of distance to inform them
# Features on larger scales can disproportionately influence the model
# We want features to be on a similar scale
# Normalizing or standardizing

# Standardization
# Subtract the mean and divide by the varianve
	# All features are centered around zero and have a variance of one

# Subtract the minimum and divide by the range
	# minimum zero and maximum one

# Normalize- chane the range of the data from -1 to 1

from sklearn.preprocessing import StandardScaler
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))

# Using a pipeline
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
knn_scaled.score(X_test, y_test)

# CV with a pipeline
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()),
				('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {"knn__n_neighbors": np.arange(1,50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridSeachCv(pipeline, param_grid = parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

print(cv.best_score_)
print(cv.bst_params_)

###### Exercises
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]
# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
#Calculate and print R-squared
print(pipeline.score(X_test, y_test))


# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)
# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)
# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)
# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)


##### Chapter 4
##### Section 4
##### Evaluating multiple Models
"""
Size of the dataset
Interpretability
Flexibility

Regression Models can be evaluated by
	Root Mean Squared Error
	R-Squared


Classification Models
	Accuracy
	Confusion Matrix
	Precision, recall, F1-score
	ROC AUC

Train Several Models and Evaluate their performance


Models that are effected by Scaling
KNN
Linear Regression
Logistic Regression
Atrificial Neural Networks


Best to scale data
"""

import matplotlib as plotfrom 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X = music.drop("genre", axis=1).values
y = music['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {"Logistic Regression": LogisticRegression(),
					"KNN": KNeighborsClassifier(),
					"Decision Tree": DecisionTreeClassifier()}
results = []
for model in models.values():
	kf = KFold(n_splits = 6, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
	results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()

for name, model in models.items():
	model.fit(X_train_scaled, y_train)
	test_score = model.score(X_test_scaled, y_test)
	print("{} Test Set Accuracy: {}".format(name, test_score))


	##### Exercise
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)  
  # Append the results
  results.append(cv_scores)
# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()



# Import mean_squared_error
from sklearn.metrics import mean_squared_error

for name, model in models.items():
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)
  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)  
  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))


#Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():  
  #Instantiate a KFold object
  kf = KFold(n_splits=6, random_state=12, shuffle=True)  
  # Perform cross-validation
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()


# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]
# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}
# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)
# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))


