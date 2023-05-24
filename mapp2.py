import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('./mgrades.csv')
data = data.dropna() # 빈값이 있으면 제거

y_train = data['grade_received'].values.reshape(-1, 1).astype('float32')
x_train = []
for i, rows in data.iterrows():
    x_train.append([float(rows['concentration']), float(rows['hours_studied'])])

print("x_train", x_train)
print("y_train", y_train)


# Define parameters
learning_rate = 0.0001
training_epochs = 1000

# Weight and bias
w = tf.Variable(tf.zeros([2,1]), name="weight", dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), name="bias")

# Linear regression (Wx + b)
def linear_regression(x):
    return tf.matmul(x, w) + b

# Mean square error
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent Optimizer
optimizer1 = tf.optimizers.SGD(learning_rate)
optimizer2 = tf.optimizers.Adam(learning_rate)

# Optimization process
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        pred = linear_regression(x_train)
        loss = mean_square(pred, y_train)

    # Compute gradients
    gradients = g.gradient(loss, [w, b])
    
    # Update W and b following gradients
    optimizer1.apply_gradients(zip(gradients, [w, b]))
    optimizer2.apply_gradients(zip(gradients, [w, b]))

# Run training for the given number of steps
for step in range(1, training_epochs + 1):
    # Run the optimization to update W and b values
    run_optimization()

# Making predictions
x_test = [[10,2.25], [10,5.5], [10,3.12],[9,2.25], [9,5.5], [9,3.12]]
y_pred = linear_regression(x_test)

print("y_pred", y_pred)
df = pd.DataFrame({'hours_studied': x_test, 'grade_received': y_pred.numpy().flatten()})
print(df)

exit()