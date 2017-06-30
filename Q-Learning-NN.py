import tensorflow as tf
import numpy as np
import tempfile
from gym.scoreboard import scoring
from gym import wrappers
import gym

tdir = tempfile.mkdtemp()
#This example is for the FrozenLake-v0 environment
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, tdir, force=True)

tf.reset_default_graph()

#Create Feed-Forward part of NN
inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32, name="Inputs1")
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01), name="Weights")
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

#Get the Loss
nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))

#Create our trainer
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#Set Learning Parameters
gamma = .90
e = 0.99
epsilon_decay = 0.999999999
num_episodes = 100000

#Lists to keep track of steps per episode and total reward
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        done = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.env.action_space.sample()
            #Get new state and reward from environment
            s1,r,done,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + gamma*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            e *= epsilon_decay
            if done == True:
                #Reduce chance of random action as we train the model.
                e *= epsilon_decay
                break
        jList.append(j)
        rList.append(rAll)

print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
print ("Score: ", scoring.score_from_local(tdir))
env.close()
gym.upload(tdir, api_key='sk_nEH5oOSpRuawpTyUep74uA ')
