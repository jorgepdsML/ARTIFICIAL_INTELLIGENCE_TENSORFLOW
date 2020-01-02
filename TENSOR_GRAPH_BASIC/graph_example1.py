#written by jorge orlando miranda ;ahui
#youtube channel : jorge miranda redes neuronales
"""
in this example , i will show you how to implement 2 basic operatons using
tensorflow graph , firts of all in tensorflow 2 , eager execution is enabled
by default (operations on tensor objects don't need graph to be executed
"""
import numpy as np
import tensorflow as tf
#disable eager execution
tf.compat.v1.disable_eager_execution()
#define 2 constants
#first constant with name X1
x1=tf.constant(5.0,dtype=tf.float32,name="X1")
#second constant with name X2
x2=tf.constant(10,dtype=tf.float32,name="X2")
#add operation
x3=tf.add(x1,x2,name="ADD")
#1 way to use session in tensorflow (it's desirable to use context manager to do that )
#create a session object to run this tensor and operation
sesion1=tf.compat.v1.Session()
#run
result=sesion1.run(x3)
#print the result
print(result)
#use use of a summary to write event files ,the graph will be displayed
#using tensorboard command in the terminal
#the first argument is the name of a directory that will save event files
#the second argument choose which graph will be launched in tensorboard
writer=tf.compat.v1.summary.FileWriter('grafo1',graph=tf.compat.v1.get_default_graph())
#close the session
sesion1.close()

