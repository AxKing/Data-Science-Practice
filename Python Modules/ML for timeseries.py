Machine Learning for Time Series Data in Python


##### Chapter 1
##### Section 1
##### Timeseries Kinds and Applications

# Time series - data that changes over time.
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
data.head()

fig, ax = plt.subplots(figsize=(12,6))
data.plot('date', 'close', ax=ax)
ax.set(title='AALP daily closing price')

# Focus of the Course
# Feature Extraction
# Model Fitting
# Prediction and validation

##### Chapter 1
##### Section 1
##### Exercises

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()

##### Chapter 1
##### Section 2
##### Machine learning basics

array.shape
array[3:]
array[:3]
df.head()

# matplotlib
fig, ax = plt.subplots()
ax.plot()

#pandas
fig, ax = plt.subplots()
df.plot(...,ax=ax)

from sklearn.svm import LinearSVC # support vector machine
# scikit-learn expects (samples, features)

.reshape()
array.reshape(-1,1).shape()
# -1 will automatically fill that axis with remaining values

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X,y)

model.coef_
predictions = model.predict(X_test)

##### Chapter 1
##### Section 2
##### Exercises

from sklearn.svm import LinearSVC
# Construct data for the model
X = data[['petal length (cm)', 'petal width (cm)']]
y = data[['target']]
# Fit the model
model = LinearSVC()
model.fit(X, y)


# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]
# Predict with the model
predictions = model.predict(X_predict)
print(predictions)
# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()


from sklearn import linear_model
# Prepare input and output DataFrames
X = housing[['MedHouseVal']]
y = housing[['AveRooms']]
# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)


# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1,1))
# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()


##### Chapter 1
##### Section 3
##### Machine Learning and Time Series Data

from glob import glob
files = glob('data/heartbeat-sounds/files/*.wav')
print(files)


import librosa as lr
audio, sfreq = lr.load('data/heartbeat-sounds/proc/files/murmur__201101051104.wav')
print(sfreq)

# creating timestamps for the data
indicies = np.arrange(0,len(audio))
time = indicies / sfreq
# or
final_time = (len(audio) - 1) / sfreq
time = np.linspace(0, final_time, sfreq)


data = pd.read_csv('path/to/data.csv')
data.columns
data.head()
df['date'].dtypes

df['date'] = pd.to_datetime(df['date'])



##### Chapter 1
##### Section 3
##### Exercises

import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()



# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data.columns:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()




##### Chapter 2
##### Section 1
##### Classification and Feature Enginering

# first visualize raw data
ixs = np.arrange(audio.shape[-1])
time = ixs / sfreq
fig, ax = plt.subplots()
ax.plot(time, audio)

# Do summary statistics
#Calculating Multiple Features
print(audio.shape)
# (n_files, time)
means = np.mean(audio, axis=-1)
maxs = np.max(audio axis = -1)
stds = np.std(audio, axis = -1)

print(means.shape)


# Import a Linear Classifier
from sklearn.svm import LinearSVC
# Note that means are reshaped to work with scikit-learn
X = np.column_stack([means, maxs, stds])
y = labels.reshape(-1,1)
model = LinearSVC()
model.fit(X,y)


# Scoring your scikit-learn model
from sklearn.metrics import accuracy_score
# Different input data
predictions = model.predict(X_test)
#Score our model with % correct
# Manually
precent_score = sum(predictions == labels_test) / len(labels_test)
#Using a sklearn scorer
percent score = accuracy_score(labels_test, predictions)


##### Chapter 2
##### Section 1
##### Exercises

fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)
# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq
# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T
# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()


# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()


from sklearn.svm import LinearSVC
# Initialize and fit the model
model = LinearSVC()
model.fit(X_train,y_train)
# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))

##### Chapter 2
##### Section 2
##### Improving features for classification

# calculating a rolling window statistic
print(audio.shape)
# (n_times, n_audio_files)

# smoothing out data by taking the mean
window_size = 50
windowed = audio.rolling(window=window_size)
audio_smooth = windowed.mean()

#auditory envelope
audio_rectified = audio.apply(np.abs)
audio_envelope = audio_rectified.rolling(50).mean()

# Calculate several features of the envelope, one per sound
envelope_mean = np.mean(audio_envelope, axis = 0)
envelope_std = np.std(audio_envelope, axis = 0)
envelope_max = np.max(audio_envelope, axis =0)

#Create our taining data for a classifier
X = np.column_stack([envelope_mean, envelope_std, envelope_max])
y = labels.reshape(-1,1)

cross_val_score
from sklearn.model_selection import cross_val_score

model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)

# Using Librosa to exract the tempo
import librosa as lr
audio_temp = lr.beat.temp(audio, sr=sfreq, hp_length=2**6, aggregate=None)


##### Chapter 2
##### Section 2
##### Exercises

# Plot the raw data first
audio.plot(figsize=(10, 5))
plt.show()
# Rectify the audio signal
audio_rectified = audio.apply(np.abs)
# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()
# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()
# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()


# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)
# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape(-1, 1)
# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))
# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)
# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)


# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape(-1, 1)
# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))




##### Chapter 2
##### Section 3
##### The spectogram

# A spectrogram is a sliding Fouier Transform Function (FFT)
# The Spectrogram is a Sliding 

# Calculating the STFT with Librosa
#import the functions we'll use for the STFT
from librosa.core import stft, amplitude_to_db
from librosa.display import specshow
import matplotlib.pyplot as plt

# calculate our STFT
HOP_LENGTH = 2**4
SIZE_WINDOW = 2**7
audio_spec = stft(audio, hop_length=HOP_LENGTH, n_fft=SIZE_WINDOW)

#Convert into devibels for visualization
spec_db = amplitude_to_db(audio_spec)

# Visuzlize
fig, ax = plt.subplots()
specshow(spec_db, sr=sfreq, x_axis='time', )


# CALCULA?TING SPECTRAL FEATURES
# Calculate the Spectral Centroid and Bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

#Display these features on top of the spectrogram
fig, ax = plt.subplots()
specshow(spec, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH, ax = ax)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha = 0.5)
plt.show()

#Combining spectral and temporal features in a classifier
centroids_all = []
bandwidths_all = []
for spec in spectrograms:
	bandwidths = lr.feature.spectral_bandwidth(S=lr.db_to_amplitude(spec))
	centroids = lr.feature.spectral_centroid(S=lr.dv_to_amplitude(spec))
	#calculate the mean spectral bandwidth
	bandwidths_all.append(np.mean(bandwidths))
	#calculate the mean spectral centroid
	centroids_all.append(np.mean(centroids))

# Create our X matrix
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths_all, centroids_all])


##### Chapter 2
##### Section 3
##### Exercises

# Import the stft function
from librosa.core import stft
# Prepare the STFT
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)

from librosa.core import amplitude_to_db
from librosa.display import specshow
# Convert into decibels
spec_db = amplitude_to_db(spec)
# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH, ax=axs[1])
plt.show()


import librosa as lr
# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

from librosa.core import amplitude_to_db
from librosa.display import specshow
# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)
# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
specshow(spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH, ax=ax)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()



# Loop through each spectrogram
bandwidths = []
centroids = []
for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)  
    centroids.append(this_mean_centroid)

# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape(-1, 1)
# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))



##### Chapter 3
##### Section 1
##### Predicting Data Over Time 

# Classification vs Regression

#Classification
classification_model.predict(X_test)
# -> array([0,1,1,0])
#Regression
regression_model.predict(X_test)
# -> array([0.2, 1.4, 3.6, 0.6])

# Correlation between variables changes over time.
# Regression tells you more, but correlation is easier to calculate


#Visualizing relationships between timeseries
fig, ax = plt.subplots(1,2)

# Make a line plot for each time series
axs[0].plot(x, c='k', lw= 3, alpha=0.2)
axs[0].plot(y)
axs[0].set(xlabel='time', title='X values = time')

# Encode time as color in a scatterplot
axs[1].scatter(x_long, y_long, c=np.arange(len(x_long)), cmap='viridis')
axs[1].set(xlabel='x', ylabel='y', title='Color = time')


# Fitting Regression Models with SciKit Learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
model.predict(X)

# Visalize predictions with scikit-learn
# Using Ridge Regression
alphas = [0.1, 1e2, 1e3]
ax.plot(y_test, color = 'k', alpha=0.3, lw=3)
for ii, alpha in enumerate(alphas):
    y_predicted = Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
    ax.plot(y_predicted, c=cmap(ii / len(alphas)))
ax.legend(['True Values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel='Time')


# Scoring regression models
# Correlation
# R^2

from sklearn.metrics import r2_score
print(r2_score(y_predicted, y_test))

##### Chapter 3
##### Section 1
##### Exercises

# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=prices.index, cmap=plt.cm.viridis, colorbar=False)
plt.show()


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Use stock symbols to extract training data
X = all_prices[['EBAY', 'NVDA', 'YHOO']]
y = all_prices[['AAPL']]
# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, shuffle=False)
# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)


# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()

##### Chapter 3
##### Section 2
##### Advanced Time Series Prediction

# Real world data is messy
# missing data and outliers

# fill in missing data with interpolation
    # First find where the missing values are
    missing = prices.isna()

    # Interpolate linearly within missing windows
    prices_interp = prices.interpolate('linear')
    # different methods in the interpolate method yield different results.

    # Plot the interpolated data in red and the data 1/missing values in black
    ax = prices_interp.plot(c='r')
    prices.plot(c='k', ax=ax, lw=2)

# Using a rolling window to Transform the data
# this method will convert each data point to represent a % change over time.

def percent_change(values):
    """calculates the % change between the last value and the mean of the previous values"""
    # Separate the last value and all previous values into variables
    previous_values = values[:1]
    last_value = values[-1]

    # Calculate the & difference between the last value
    # and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) \
    / np.mean(previous_values)

    return percent_change

# Applying the percentage change to the data

# Plot the raw data
fig, axs = plt.subplots(1,2, figsize=(10,5))
ax = proces.plot(ax=axs[0])

# Calculate % change and plot
ax = prices.rolling(window=20).aggregate(percent_change).plot(ax=axs[1])
ax.legend_.set_visible(False)

# Finding outliers in the data 
fig, ax = plt.subplots(1,2, figsize=(10,5))
for data, ax in zip([prices, prices_perc_change], axs):
    # Calculate the mean / standard deviation for the data
    this_mean = data.mean()
    this_std = data.std()

    # Plot the data, with a window that is 3 standard deviations around the mean
    data.plot(ax=ax)
    ax.axhline(this_mean + this_std * 3, ls ='--', c='r')
    ax.axhline(this_mean - this_std * 3, ls ='--', c='r')


# Replacing outliers with the median of the remaining values
    # Center the data so the mean is 0
    prices_outlier_centered = prices_outlier_perc - prices_outlier_perc.mean()

    # Calculate standard deviation
    std = prices_outlier_perc.std()

    # Use the absolute value of each datapoint to make it easier to find outliers
    outliers = np.abs(prices_outlier_centered) > (std * 3)

    # Replace outliers with the median value
    # We'll use np.nanmean since there may be nans around the outliers
    prices_outlier_fixed = prices_outlier_centered.copy()
    prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed)

    # Now we can plot the data 
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    prices_outliers_centered.plot(ax=axs[0])
    prices_outlier_fixed.plot(ax=axs[1])


##### Chapter 3
##### Section 2
##### Exercises

# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()
# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)


# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):
    # Create a boolean mask for missing values
    missing_values = prices.isna()
    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)
    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()

# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)

# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)


# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()


def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()


##### Chapter 3
##### Section 3
##### Creating features over time

# Uing the .aggregate for feature extraction
    # Visualize the raw data
    print(prices.head(3))

    # Calculate a rolling window, then extract two features
    feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
    print(feats.head(3))

# Using .partial()
    # if we just take the mean, it returns a single value
    a = np.array([[0,1,2], [0,1,2], [0,1,2]])
    print(np.mean(a))

    # We can use the partial function to initialize np.mean
    # with an axis parameter
    from functools import partial
    mean_over_first_axis = partial(np.mean, axis=0)

    print(mean_over_first_axis(a))

# Percentile function
print(np.percentile(np.linspace(0,200), q=20))

# Combining the percentile function with the partial function

    data = np.linspace(0,100)

    # Create a list of functions using list comprehension
    percentile_funcs = [partial(np.percentile, q=ii) for ii in [20,40,60]]

    # Calculate the output of each function in the same way
    percentiles = [i_func(data) for i_func in percentile_funcs]
    print(percentiles)

    # Calculate multiple percentiles of a rolling window
    data.rolling(20).aggregate(percentiles)


# Calculating "date-based" features
    
    # Ensure our index is a datetime
    prices.index = pd.to_datetime(prices.index)

    # Extract datetime features
    day_of_week_num = prices.index.weekday
    print(day_of_week_num[:10])

    day_of_week = prices.index.weekday_name
    print(day_of_week[:10])


##### Chapter 3
##### Section 3
##### Exercises


# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean , np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()



# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()


# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.weekday
prices_perc['week_of_year'] = prices_perc.index.week
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)



##### Chapter 4
##### Section 1
##### Creating Features from the Past

print(df)

# shift a DataFrame/Series by 3 index values towards the past.
print(df.shift(3))

#Data is a pandas Series containing time series data
data = pd.Series(...)

#shifts
shifts = [0,1,2,3,4,5,6,7]

# Create a dictionary of time-shifted data
many_shifts = {'lag_{}'.format(ii): data.shift(ii) for ii in shifts}

# Convert them into a data frame
many_shifts = pd.DaraFrame(many_shifts)

# now fit a sci-kit learn regression model
model = Ridge()
model.fit(many_shifts, data)

# Vizualize the fit model coefficients
fig,  ax = plt.subplots()
ax.bar(many_shifts.columns, model.coef_)
ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

# Set Formatting so it looks nice
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignmeent='right')


##### Chapter 4
##### Section 1
##### Exercises

# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()


# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)


def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax


# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_ , prices_perc_shifted.columns, ax=axs[1])
plt.show()




##### Chapter 4
##### Section 2
##### Cross Validating time series data

# Iterating over the "split" method yields train/test indicies
for tr, tt in cv.split(X,y):
    model.fit(X[tr],y[tr])
    model.score(X[tt],y[tt])

# For K-Fold
    from sklearn.model_selection imprt KFold
    cv = KFold(n_splits = 5)
    for tr, tt in cv.split(X,y):
        ...

# Visuzalizing model predictions
fig, ax = plt.subplots(2,1)

# Plot the indices chosen for validation on each loop
axs[0].scatter(tt, [0] * len(tt), marker ='_', s=2, lw=40)
axs[0].set(ylim=[-.1, .1], title='Test set indices (color = CV loop)', xlabel = 'Index of raw data')

# Plot the model predictions on each iteration
ax[1].plot(model.predict(X[tt]))
ax[1].set(title='Test set predictions on each CV loop', xlabel='Prediction index')


# Shuffling your data
    from sklearn.model_selection import ShuffleSplit

    cv = ShuffleSplit(n_splits=3)
    for tr, tt in cv.split(X,y):
        ...

# Time Series Data CV Iterator

from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits = 10)

fig, ax = plt.subplots(figsize=(10,5))
for ii, (tr, tt) in enumerate(cv.split(X,y)):
    # Plot training and test indicies
    l1 = ax.scatter(tr, [ii] * len(tr), c = [plt.cm.coolwarm(.1)], marker = '-', lw = 6)
    l2 = ax.scatter(tt, [ii]* len(tt), c=[plt.cm.coolwarm(.9)], marker='-', lw=6)
    ax.set(ylim=[10,-1], title='TimeSeriesSplit behavior', xlabel = 'data index', ylabel = 'CV Iteration')
    ax.legend([l1,l2], ['Training', 'Validation'])

def myfunction(estimator, X, y):
    y_pred = estimator.predict(X)
    my_custom_score = my_cutom_function(y_pred, y)
    return my_custom_score


# A Custom Correlation function
def my_pearsonr( est, X, y):
    # Generate predictions and convert to a vector
    y_pred = est.predict(X).squeeze()

    #use the numpy "corrcoef" function to calculate a correlation matrix
    my_correcoef_matrix = np.corrcoef(y_pred, y.squeeze())

    # Return a single correlation value from the matrix
    my_corrcoef = my_corrcoef_matrix[1,0]

    return my_corrcoef




##### Chapter 4
##### Section 2
##### Exercises

# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits = 10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)


# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr],y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)


# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()



##### Chapter 4
##### Section 3
##### Stationarity and Stability

# Bootstrapping the mean
from sklearn.utils import resample

# cv_coefficients has shape (n_cv_folds, n_coefficients)
n_boots = 100
bootstrap_means = np.zeros(n_boots, n_coefficients)
for ii in range(n_boots):
    # Generate Random indicies for our data with replacement,
    # then take the sample mean
    random_sample = resample(cv_coefficients)
    bootstrap_means[ii] = random_sample.mean(axis=0)

# compute the percentiles of choice for the bootstrapped means
percentiles = np.percentile(bootstrap_means, (2.5, 97.5), axis=0)



# Plot the 95% confidence intervals we just calculated 
fig, ax = plt.subplots()
ax.scatter(many_shifts.columns, percentiles[0], marker='_', s=200)
ax.scatter(many_shifts.columns, percentiles[1], marker='_', s=200)



# Stationary and Stability
# Model Performance over time
def my_corrcoef(est,X,y):
    """Return the correlation coefficient between model predictions and a 
    validation data set."""
    return np.corrcoef(y,est.predict(X))[1,0]

# Grab the data of the first index of each validation set
first_indicies = [data.index[tt[0]] for tr, tt in cv.split(X, y)]

# Calculate the CV scores and convert to a Pandas Series
cv_scores = cross_val_score(model, X, y, cv = cv, scoreing = my_corrcoef)
cv_scores = pd.Series(cv_scores, index = first_indicies)



# Visualizing model scores as a timeseries
fig, ax = plt.subplots(2,1, figsize = (10,5), sharex = True)

# Calculate a rolling mean of scores over time
cv_scores_mean = cv_scores.rolling(10, min_periods=1).mean()
cv_scores.plot(ax=axs[0])
axs[0].set(title='Validation scores (correlation)', ylim=[0,1])

# plot the raw data
data.plot(ax=axs[1])
axs[1].set(title='Validation data')


# Restrict the size of the training window
# only keep 100 datapoints in the training data
window = 100

# Initialize the CS with this window size
cv = TimeSeriesSplit(n_splits=10, max_train_size=window)




##### Chapter 4
##### Section 3
##### Exercises

from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles



# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_



# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()



from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))


# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()



# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores


# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()





