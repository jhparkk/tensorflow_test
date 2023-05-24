import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

# Define parameters
learning_rate = 0.01
training_epochs = 1000

# Weight and bias
W = tf.Variable(tf.zeros([1,1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# Linear regression (Wx + b)
def linear_regression(x):
    return tf.matmul(x, W) + b

# Mean square error
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        pred = linear_regression(X_train)
        loss = mean_square(pred, y_train)

    # Compute gradients
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Run training for the given number of steps
for step in range(1, training_epochs + 1):
    # Run the optimization to update W and b values
    run_optimization()

# Making predictions
y_pred = linear_regression(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.numpy().flatten()})
print(df)
