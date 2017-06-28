import tensorflow as tf
import numpy as np
import os
#Only show ERRORS
tf.logging.set_verbosity(tf.logging.ERROR)
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#An estimator that does linear regression
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

#Setup training data
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

#Setup evaluation data
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#Setup input, this handes what data we input, how much, and how many "runs" to do
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size= 4, num_epochs=1000)
input_fn_eval = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size= 4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

#Let's see how we did!
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=input_fn_eval)

#Print out what we got
print("Train Loss: %r"%train_loss)
print("Eval Loss: %r"%eval_loss)
