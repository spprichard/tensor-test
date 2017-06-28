import os
import tensorflow as tf
#Hides debug unwanted debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Model Parameters
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.1], dtype=tf.float32)

#Input
x = tf.placeholder(dtype=tf.float32)

#Linear Model y = Wx + b
linear_model = W*x + b

#Output
y = tf.placeholder(tf.float32)

#Loss
loss = tf.reduce_sum(tf.square(linear_model - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
#Init/Reset all variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#Training
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

#Evaluate Model: Evaluate in training data
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y:y_train})
print("W: %s, b: %s, Loss: %s"%(curr_W, curr_b, curr_loss))
