# TensorFlow.py

### Chapter 1
### Section 1
### Constants and Variables

import tensorflow as tf

# 0D Tensor
d0 = tf.ones((1, ))

#1D Tensor
d1 = tf.ones((2, ))

#2D Tensor
d2 = tf.ones((2, 2))

#3D Tensor
d3 = tf.ones((2, 2, 2))

#Print the 3D tensor
print(d3.numpy())

from tensorflow import constant

#Define a 2x3 constant.
a = constant(3, shape=[2,3])

b = constant([1,2,3,4], shape = [2,2])


# Using Convenience functions to define constants
tf.constant()
constant([1,2,3])

tf.zeros()
zeros([2,2])

tf.zeros_like()
zeros_like(input_tensor)

tf.ones()
ones([2,2])

tf.ones_like()
ones_like(input_tensor)

tf.fill()
fill([3,3], 7)

# Define a variable
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype = tf.float32)
a1 = tf.Variable([1, 2, 3, 4, 5, 6], dtype = tf.int16)

# Define a constant
b = tf.constant(2, tf.float32)

c0 = tf.multiply(a0, b)
c1 = a0 * b

### Chapter 1
### Section 1
### Exercises

# Import constant from TensorFlow
from tensorflow import constant
# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)
# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)
# Print constant shape
print('\n The shape is:', credit_constant.shape)


# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])
# Print the variable A1
print('\n A1: ', A1)
# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()
# Print B1
print('\n B1: ', B1)



### Chapter 1
### Section 2
### Basic Operations

# Addition
#Import constand and add from tensorflow
from tensorflow import constant, add

# Define 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])

# Define 1-Dimensional tensors
A1 = constant([1,2])
B1 = constant([3,4])

# Define 2-dimensional tensors
A2 = constant([[1,2], [3,4]])
B2 = constant([[5,6], [7,8]])

# Perform tensor addition with add()
C0 = add(A0, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)

add() # element wise addition
multiply() # element wise multiplication 
matmul() # matrix multiplication

# Import operators from tensorflow
from tensorflow import ones, matmul, multiply

# Define tensors
A0 = ones(1)
A31 = ones([3,1])
A34 = ones([3,4])
A43 = ones([4,3])

# what operations are valid?
multiply(A0, A0), multiply(A31, A31) and multiply(A34, A34)
matmul(A43, A34), but not matmul(A43, A43)


reduce_sum() #operator sums over the dimensions of a tensor
reduce_sum(A) # sums over all dimensions of A
reduce_sum(A, i) #sums over dimension i

from tensorflow import ones, reduce_sum
A = ones([2,3,4]) # this is a 2 x 3 x 4 tensor that consists of ones.

B = reduce_sum(A) # 2x3x4 = 24
B0 = reduce_sum(A, 0) # 3X4 matrix of 2s
B1 = reduce_sum(A, 1) # 2X4 matrix of 3s
B2 = reduce_sum(A, 2) # 2x3 matrix of 4s

### Chapter 1
### Section 2
### Exercises

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])
# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)
# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)
# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))


# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])
# Compute billpred using features and params
billpred = matmul(features, params)
# Compute and print the error
error = bill - billpred
print(error.numpy())

### Chapter 1
### Section 3
### Advanced Operations

gradient() 
x = tf.Variable(-1.0)
with tf.GradientTape() as tape:
	tape.watch(x)
	y = tf.multiply(x, x)
g = tape.gradient(y,x) 
print(g.numpy())

reshape()
gray = tf.random.uniform([2,2], maxval=255, dtype='int32')
grey = tf.reshape(gray, [2*2, 1])

color = tf.random.uniform([2,2,3], maxval=255, dtype='int32')
color = tf.reshape(color, [2*2, 3])





### Chapter 1
### Section 3
### Exercises

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, [28*28, 1])
# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, [28*28*3, 1])


def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))


# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))
# Multiply letter by model
output = matmul(letter, model)
# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output, 0)
print(prediction.numpy())


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

### Chapter 2
### Section 1
### Input Data

pd.read_csv('')
np.array()

read_csv()
filepath_or_buffer = accepts a file path or URL.
sep = delimeter between columns.
delim_whitespace = boolean for whether to delimit whitespace.
encoding = specifies encoding to be used if any.

# Setting the data type
# Convert price column to float32
price = np.array(housing['price'], np.float32)
# OR
price = tf.cast(housing['price'], tf.float32)
#convert waterfront column to Boolean
waterfront = np.array(housing['waterfront'], np.bool)
# OR
waterfront = tf.cast(housing['waterfront'], tf.bool)



### Chapter 2
### Section 1
### Exercises

# Import pandas under the alias pd
import pandas as pd
# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'
# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)
# Print the price column of housing
print(housing['price'])


# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf
# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)
# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)
# Print price and waterfront
print(price)
print(waterfront)




### Chapter 2
### Section 2
### Loss Functions

higher value -> worse fit
parameters should minimize the loss function

Common loss functions

Mean Squared Error MSE
tf.keras.losses.mse()
# Strongly penalizes outlies
# high sensitivity near the minimum


Mean Absolute Error MAE
tf.keras.losses.mae()
# Scales linearly with size of error
# Low Sensitivity near minimum


Huber Error
tf.keras.losses.Huber()
# High sensitivity near the minimum
# Scales linearly with size of error

# Compute loss
tf.keras.losses.mse(targets, predictions)

def linear_regression(intercept, slope = slope, features= features):
	return intercept + features * slope

def loss_function(intercept, slope, targets= targets, features=features):
	predictions = linear_regression(intercept, slope)

	return tf.keras.losses.mse(targets, predictions)

loss_function(intercept, slope, test_targets, test_features)
loss_function(intercept, slope)


### Chapter 2
### Section 2
### Exercises

# Import the keras module from tensorflow
from tensorflow import keras
# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)
# Print the mean squared error (mse)
print(loss.numpy())


# Import the keras module from tensorflow
from tensorflow import keras
# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)
# Print the mean absolute error (mae)
print(loss.numpy())


# Initialize a variable named scalar
scalar = Variable(1.0, float32)
# Define the model
def model(scalar, features = features):
  	return scalar * features
# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)
# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())


### Chapter 2
### Section 3
### Linear Regression

# Define the targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

# Define the intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.(0.1, np.float32)

def linear_regression(intercept, slope, features = size ):
	return intercept + features * slope

def loss_function(intercept, slope, targets = price, features = size):
	predictions = linear_regression(intercept, slope)
	return tf.keras.losses.mse(targets, predictions)

opt = tf.keras.optimizers.Adam()

for j in range(1000):
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
	print(loss_function(intercept, slope))

print(intercept.numpy(), slope.numpy())

### Chapter 2
### Section 3
### Exercises

# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + slope*features
# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)
# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())


# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)
for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())
# Plot data and regression line
plot_results(intercept, slope)


# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]
# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)
# Define the optimize operation
opt = keras.optimizers.Adam()
# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)

### Chapter 2
### Section 4
### Batch Training

for batch in pd.read_csv('kc_housing.csv', chunksize=100):
	price = np.array(batch['price'], np.float32)

	size = np.array(batch['size'], np.float32)


intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

def linear_regression(intercept, slope, features):
	return intercept + features * slope

def loss_function(intercept, slope, targets, features):
	predictions = linear_regression(intercept, slope, features)
	return tf.keras.losses.mse(targets, predictions)

opt = tf.keras.optimizers.Adam()

for batch in pd.read_csv('kc_housing.csv', chunksize=100):
	price_batch = np.array(batch['price'], np.float32)
	size_batch = np.array(batch['lot_size'], np.float32)
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

print(intercept.numpy(), slope.numpy())

### Chapter 2
### Section 4
### Exercises

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)
# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + slope * features
# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)


# Initialize Adam optimizer
opt = keras.optimizers.Adam()
# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)
	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)
	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])
# Print trained parameters
print(intercept.numpy(), slope.numpy())


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

### Chapter 3
### Section 1
### Dense Layers

# define inputs (features)
inputs = tf.constant([1,35])

# Define weights
weights = tf.Variable([[-0.05], [-0.01]])

# Define the bias
bias = tf.Variable([0.5])

# Low Level Approach
# a simple dense layer
# multiply inputs (features) by the weights
product = tf.matmul(inputs, weights)
#Define dense layer
dense = tf.keras.activations.sigmoid(product+bias)


# High-Level Approach
#Defining a complete Model
inputs = tf.constant(data, tf.float32)
# Define the first layer 10-nodes,
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)
# Define second dense layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)
# Define outputs (Predictions layer)
outputs = tf.keras.layers.Dense(1,activation='sigmoig')(dense2)


### Chapter 3
### Section 1
### Exercises

# Initialize bias1
bias1 = Variable(1.0)
# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))
print(weights1)
# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)
# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)
# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))


# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)
# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))
# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)
# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')


# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)
# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1+bias1)
# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)


# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)
# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)
# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)
# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


### Chapter 3
### Section 2
### Activation Functions

sigmoid - binary classification
tf.keraas.activations.sigmoid
sigmoid

relu - rectified Linear Unit - used in the inner layers
tf.keras.activations.relu()

softmax - output layer (>2 classes)
tf.keras.activations.softmax()

#High Level
inputs = tf.constant(borrower_features, tf.float32)
dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)
dense2 = tf.keras.laters.Dense(8, activation='sigmoid')(dense1)
outputs = tf.keras.layers.Dense(4, activation='softmax')(dense2)




### Chapter 3
### Section 2
### Exercises

# Construct input layer from features
inputs = constant(bill_amounts, float32)
# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)
# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)
# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)
# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)


# Construct input layer from borrower features
inputs = constant(borrower_features, float32)
# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)
# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)
# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)
# Print first five predictions
print(outputs.numpy()[:5])



### Chapter 3
### Section 3
### Optimizers

# Finding a set of weights that corresponds with the minimal loss function

#Stochastic Gradient Descent (SGD)
tf.keras.optimizers.SGD()
learning_rate =

#Root Mean Squared (RMS)RMSE
tf.keras.optimizers.RMSprop()
learning_rate = different for each feature
momentum = 
decay = low value prevents momentum to accumulate

The adam optimizer
Adaptive Moment Optimizer (adam)
tf.keras.optimizers.Adam()
learning_rate = 
beta1 = momentum will decay faster


def model(bias, weights, features = borrower_features):
	product = tf.matmul(features, weights)
	return tf.keras.activations.sigmoid(product+bias)

def loss_function(bias, weights, targets = default, features = borrower_features):
	predictions = model(bias, weights)
	return tf.keras.losses.binary_crossentropy(targets, predictions)

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights), var_list=[bias, weights])




### Chapter 3
### Section 3
### Exercises

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)
# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)
for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])
# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)
# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)
for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])
# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


### Chapter 3
### Section 4
### Training a Network in Tensorflow

# Initializing variables in TensorFlow
# low level
weights = tf.Variable(tf.random.normal([500,500]))

weights = tf.Variable(tf.random.truncated_normal([500,500]))

# High Level
dense = tf.keras.layers.Dense(32, activation='relu')

dense = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='zeros')

# Dropout is a method to help prevent overfitting
inputs = np.array(borrow_features, np.float32)
dense1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
dense2 = tf.keras.Dense(16, activation='relu')(dense1)
dropout1 = tf.keras.layers.Dropout(0.25)(dense2) #Drops 25% of nodes randomly
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)



### Chapter 3
### Section 4
### Exercises


# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))
# Initialize the layer 1 bias
b1 = Variable(ones([7]))
# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))
# Define the layer 2 bias
b2 = Variable(0.0)

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)
# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])
# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)
# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])
# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)
# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

### Chapter 4
### Section 1
### Defining neural networks with keras

#Building a sequential model
	#Define the model
model = keras.Sequential()
	#Adding the first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28, )))
	# second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))
	# Output layer
model.add(keras.layers.Dense(4, activation = 'softmax'))
	# Compile the model
model.compile('adam', loss = 'categorical_crossentropy')
	# summarize the model
print(model.summary())


# Using multiple models for predictions
	# Model Input 1
model1_inputs = tf.keras.Input(shape(28*28,))
	# Model Input 2
model2_inputs = tf.keras.Input(shape=(10,))

	# Layer 1 for Model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)
	# layer 2 for Model 1
model1_layer2 = tf.keras.layers.Dense(4, activation = 'softmax')(model1_layer1)
	# layer 1 for Model 2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)
	# layer 2 for Model 2
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1)

	# Merge Model 1 and Model 2
merged = tf.keras.layers.add([model1_layer2, model2_layer2])

	#Define a functional model
model = tf.keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)
# compile the model
model.compile('adam', loss='categorical_crossentropy')





### Chapter 4
### Section 1
### Exercises

# Define a Keras sequential model
model = keras.Sequential()
# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))
# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))
# Define the output layer
model.add(keras.layers.Dense(4, activation = 'softmax'))
# Print the model architecture
print(model.summary())


# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape =(784,)))
# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))
# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))
# Compile the model
model.compile('adam', loss='categorical_crossentropy')
# Print a model summary
print(model.summary())


# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)
# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)
# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)
# Print a model summary
print(model.summary())



### Chapter 4
### Section 2
### Training with Keras

1. load and clean the data
2. Define the model
3. Train and Validate the model
4. Evaluate the model

import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(4, activation = 'softmax'))
model.compile('adam', loss='categorical_crossentropy')

model.fit(image_features, image_labels)
	batch_size = examples in each batch
	epochs = number of times you train over all of the batches
	validation_split = divides the dataset into train and validate, 0.20 gives 20% data to validate

# Changing the metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=10, validation_split=0.20)

model.evaluation(test)


### Chapter 4
### Section 2
### Exercises

# Define a sequential model
model = keras.Sequential()
# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))
# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))
# Compile the model
model.compile('SGD', loss='categorical_crossentropy')
# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)



# Define sequential model
model = keras.Sequential()
# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))
# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))
# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.10)


# Define sequential model
model = keras.Sequential()
# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))
# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))
# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])
# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.50)


# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)
# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)
# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)
# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)
# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))


### Chapter 4
### Section 3
### Training Models with Estimators API

1. Define Feature columns
2. load and transform data
3. define an estimator
4. apply train operation

1. Define feature columns
	#numeric
size = tf.feature_column.numeric_column("size")
	#categorical
rooms = tf.feature_column.categorical_column_with_vocabulary_list('rooms', ['1', '2', '3', '4', '5'])

features_list = [size, rooms]
# -- features_list = [tf.feature_column.numeric_column('image', shape=(784,))]
2. load and transform data 
def input_fun():
	features = {'size': [1340, 1690, 2720], "rooms": [1,3,4]}
	labels = [221900, 538000, 180000]
	return features, labels

3. Define an estimator
model0 = tf.estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[10,6,6,3])
model1 = tf.estimator.DNNClassifier(feature_columns=feature_list, hidden_units = [32,16,8], n_classes=4)

4. apply training operation
model0.train(input_fn, steps=20)
model1.train(input_fn, steps=20)


### Chapter 4
### Section 3
### Exercises


# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")
# Define the list of feature columns
feature_list = [bedrooms, bathrooms]
def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)


