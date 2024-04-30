# -*- coding: utf-8 -*-
"""
Created on Sep 2023

@author: Kaike Alves
"""

# Import libraries
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from permetrics.regression import RegressionMetric
import statistics as st
import matplotlib.pyplot as plt

# Neural Network
from tensorflow import keras
from keras.utils import plot_model
# Optimize Network
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from MackeyGlassGenerator import MackeyGlass


#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "MackeyGlass"

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X, y = Create_Leg(MG, ncols = 4, leg = 6, leg_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]

# Spliting the data into train and test
n = X_train.shape[0]
val_size = round(n*0.8)
X_train, X_val = X[:val_size,:], X[val_size:,:]
y_train, y_val = y[:val_size], y[val_size:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape the inputs
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()

# -----------------------------------------------------------------------------
# Executing the Random Search for the MackeyGlass time series
# -----------------------------------------------------------------------------

Model = "WaveNet"

# -----------------------------------------------------------------------------
# Optimize hyper-parameters
# -----------------------------------------------------------------------------

# Define the function to create models for the optimization method
def build_model(dilation_rate=(1, 2, 4, 8), repeat=2, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[X_train.shape[1],X_train.shape[2]]))
    for rate in dilation_rate * repeat:
        model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


# Wrapper around eras model built
keras_reg = KerasRegressor(model=build_model, dilation_rate=(1, 2, 4, 8), repeat=2, learning_rate=3e-3)

# Random search options
param_distribs = {
    "dilation_rate": [(1),(1,2),(1,2,4),(1,2,4,8),(1,2,4,8,16),(1,2,4,8,16,32)],
    "repeat": [1,2],
    "learning_rate": reciprocal(1e-5,0.5)}

# Call the Random Search function
rnd_search_cv = RandomizedSearchCV(estimator=keras_reg, param_distributions=param_distribs, n_iter=100, cv = 3)
# Start the optimization
rnd_search_cv.fit(X_train, y_train, epochs = 100, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=2)])

# Print the best model parameters
print(f'\nBest parameters:\n {rnd_search_cv.best_params_}')

# Print the best model score
print(f'\nBest score:\n {rnd_search_cv.best_score_}\n\n')

# Implement the prediction method
y_pred = rnd_search_cv.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Serie}.eps', format='eps', dpi=1200)
plt.show()


# -----------------------------------------------------------------------------
# Run and save the network for the best hyper-parameters
# -----------------------------------------------------------------------------

# Define the neural network
model = build_model(**rnd_search_cv.best_params_)

# Checkpoint functions to recover the best model
checkpoint_cb = keras.callbacks.ModelCheckpoint(f'RandomSearchResults/{Serie}.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs = 100, validation_data=(X_val, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Implement the prediction method
y_pred = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)

# Results
WaveNet = f'{Model} & {NRMSE:.5f} & {NDEI:.5f} & {MAPE*100:.2f} & -'
print(f"\n{WaveNet}")

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Serie}.eps', format='eps', dpi=1200)
plt.show()

# Print the summary of the model
print(model.summary())

# Plot the model architeture
# You must install pydot (`pip install pydot`) and install graphviz (https://graphviz.gitlab.io/download/).
plot_model(model, to_file=f'ModelArchiteture/{Serie}.png', show_shapes=True, show_layer_names=True)

