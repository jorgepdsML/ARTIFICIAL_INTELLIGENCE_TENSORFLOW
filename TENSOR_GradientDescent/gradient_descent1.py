"""
in this file  , I will show you how to to use GradientDescentOptimizer
using a basic example , I will build a square loss function and apply
the gradient descent algorithm to this loss function.

"""
import numpy as np
import tensorflow as tf
#disable eager execution
tf.compat.v1.disable_eager_execution()
#define a tensor variable
#initial value
init_value=10.2
w=tf.Variable(init_value,dtype=tf.float32,name="W")
#optimizer , square of a variable
losf=tf.square(w)
#GradientDescentOptimizer(learnig rate)
opt=tf.compat.v1.train.GradientDescentOptimizer(0.3)
#minimize the loss function
train=opt.minimize(loss=losf)
#create a session
with tf.compat.v1.Session() as sesion1:
    #initialize global variables
    sesion1.run(tf.compat.v1.global_variables_initializer())
    print(sesion1.run(w))
    for a in range(15):
        #show the W variable at each iteration
        sesion1.run(train)
        print("W at iteration {} is : ".format(a),sesion1.run(w))

