import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Example data
data = {'Hours_Studied': [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4],
        'Grades': [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69]
        }

df = pd.DataFrame(data)

# Split data into input X and output y
X = df['Hours_Studied'].values.reshape(-1,1).astype('float32')
y = df['Grades'].values.reshape(-1,1).astype('float32')

# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model using Keras
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Making predictions
print("X_test Type: ", type(X_test))
print("X_test: ", X_test)
y_pred = model.predict(X_test)

print("y_pred: ", y_pred)

print(" len x_train: ", len(X_train))
print(" len y_train: ", len(y_train))
print(" len x_test: ", len(X_test))
print(" len y_test: ", len(y_test))
print(" flatten len y_test: ", len(y_test.flatten()), "y_pred: ", len(y_pred.flatten()))

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)