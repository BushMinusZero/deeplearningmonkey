"""
Hello World Neural Network
- train a neural network to fit points on a line
"""
import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Simple Example
# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Random Numbers
xs = np.random.normal(size=20)
ys = 2*xs-1 # <- learn this linear function

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
