"""
written by jorge miranda ñahui
in this file  , I will show you how to to use GradientDescentOptimizer
using a basic example , I will build a square loss function and then apply
the gradient descent algorithm to this loss function.
keep in mind that , the gradiente descent algorithm will be computed based on the independent variable
of the your loss function. 

"""
import numpy as np
import tensorflow as tf
#disable eager execution
tf.compat.v1.disable_eager_execution()
#define a tensor variable
#initial value
init_value=10.2
#create a tensor variable with a initial value and 32 bits (float) data type 
w=tf.Variable(init_value,dtype=tf.float32,name="W")
#loss function
losf=tf.square(w)
#creation of object for the use of gradient descent algorithm
#GradientDescentOptimizer(learnig rate)
opt=tf.compat.v1.train.GradientDescentOptimizer(0.3)
#minimize the loss function 
#this method or function internally calls two methods (compute_gradiendts and then apply_gradients) of 
#an GradientDescentOptimizer Object
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

