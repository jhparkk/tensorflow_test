import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('./mgrades.csv')
data = data.dropna() # 빈값이 있으면 제거


# 모델 지정 및 컴파일
#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.000005), loss='mean_squared_error', metrics=['accuracy']) 
#model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.000005), loss='mean_squared_error', metrics=['accuracy']) 

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1, activation='relu'), # 레이어1 (node count=64, 활성함수=relu)
# ])
# model.compile(optimizer='sgd', loss='mean_squared_error')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')


y_train = data['grade_received'].values
x_train = []
for i, rows in data.iterrows():
    x_train.append([rows['concentration'], rows['hours_studied']])

print("shape x_train", np.array(x_train).shape)
print("x_train", x_train)
print("y_train", y_train)

model.fit(np.array(x_train), np.array(y_train), epochs=1000, verbose=0)  # 학습 횟수 1000회 - w 값을 도출


x_test = [[10,2.25], [10,5.5], [10,3.12],[9,2.25], [9,5.5], [9,3.12]]
print("x_test type : ", type(x_test), "np.array(x_test) type : ", type(np.array(x_test)))
print("x_test : ", x_test)
y_pred = model.predict(np.array(x_test))
print("y_pred", y_pred.flatten())
print("len x_train : ", len(x_train))
print("len y_train : ", len(y_train))
print("len x_test : ", len(x_test))
print("len y_pred : ", len(y_pred.flatten()))

df = pd.DataFrame({'hours_studied': x_test, 'grade_received': y_pred.flatten()})
print(df)

exit()