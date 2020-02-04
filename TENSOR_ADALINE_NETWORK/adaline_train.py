"""
written by jorge orlando miranda ñahui
visit and suscribe to my youtube's channel : jorge miranda redes neuronales
github repo :
https://github.com/jorgepdsML
INTELLIGENCE ARTIFICIAL

ADAptive LInear NEuron (ADALINE)
to built the ADALINE arquitecture using tensorflow , we need to
specify if we want to disable the eager execution mode or not
1 - by default eager execution is enabled (it does not need session to run graph)
2 -using tf.compat.v1.disable_eager_execution() , we will use graph as in
tensorflow 1.0 . it means that to execute a graph we need
a session object to run it
"""
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow as tf
#disble eager execution
tf.compat.v1.disable_eager_execution()
print("IS EAGER MODE ENABLED? : ",tf.executing_eagerly())

#--------TRAINING DATA POINTS
Xi=np.array([[-1,-1],
             [-1,1],
             [1,-1],
             [1,1]],dtype=float)
Ti=np.array([-1,
             -1,
             -1,
             1],dtype=float)
#Number of samples
N=np.size(Ti)
#------------
#use of a tf.Graph class to build a customized graph
#---------creating a Graph object -------------------------
graf1=tf.Graph()
#use of its method as_default() to make the graf1 be the default graph
#use of context manager to add operations to the default graph1
#----------    add operations within this context manager (graph)---
with graf1.as_default():
    #Number of inputs=2
    R=2
    #Number of neurons=1
    S=1
    #---------------------------INPUT AND OUTPUT----------------
    #INPUT Rx1
    X=tf.compat.v1.placeholder(shape=(R,1),dtype=tf.float64,name="X")
    #OUTPUT SHAPE Sx1
    T=tf.compat.v1.placeholder(shape=(S,1),dtype=tf.float64,name="T")
    #------------------------SYNAPTIC WEIGHTS AND BIASES
    # Variables to define the weights and bias
    #WEIGHTS SxR
    W = tf.Variable(np.random.randn(S,R), dtype=tf.float64, name="W")
    #BIASES Sx1
    B = tf.Variable(np.random.randn(S,1), dtype=tf.float64, name="B")
    #---------------------------------------------------
    #COMPUTE THE OUTPUT OF W*X
    Z1=tf.matmul(W,X,name="Z1")
    #COMPUTE THE OUPUT OF B*1
    Z2=tf.matmul(B,tf.ones(shape=(1,1),dtype=tf.float64),name="Z2")
    #ADD OPERATION Z1+Z2
    V=tf.add(Z1,Z2,name="NET_INPUT")
    #LINEAR TRANSFER FUNCTION
    Y=tf.identity(V,name="LINEAR")
    #Y=tf.nn.relu(tf.sign(V))
    #ERROR
    E=tf.subtract(T,Y,name="ERROR")
    #COST FUNCTION
    #loss=tf.reduce_mean(tf.square(E)/2,name="COST_FUNCTION")
    lr=0.9
    #deltaw=lr*tf.matmul(E,tf.transpose(X))
    #deltab=lr*E
    #------------LEARNING RULE------------
    #---------SYNAPTIC WEIGHTS
    W_up=tf.compat.v1.assign(W,W+lr*tf.matmul(E,tf.transpose(X)),name="WEIGHTS_UPDATE")
    #------------BIAS ---------------
    B_up=tf.compat.v1.assign(B,B+lr*E,name="BIASES_UPDATE")
#use of session  , tensorflow 1.0 STYLE
with tf.compat.v1.Session(graph=graf1) as sesion1:
    sesion1.run(tf.compat.v1.global_variables_initializer())
    epochs=10
    for epoc in range(epochs):
        # iterate over all the examples
        for id in range(N):
            #lossv = sesion1.run(loss, feed_dict={X: np.reshape(Xi[id, :], (2, 1)), T: np.reshape(Ti[id], (1, 1))})
            o1 = sesion1.run(W_up, feed_dict={X: np.reshape(Xi[id, :], (2, 1)), T: np.reshape(Ti[id], (1, 1))})
            b1 = sesion1.run(B_up, feed_dict={X: np.reshape(Xi[id, :], (2, 1)),T: np.reshape(Ti[id], (1, 1))})

        print("WEIGTHS AFTER EPOCH:", epoc + 1, "IS: ", sesion1.run(W))
        print("BIAS AFTER EPOCH:", epoc + 1, "IS: ", sesion1.run(B))
#PLOT THE DECISION SURFACE AND THE TRAINING DATA POINTS
from matplotlib import pyplot as plt
# define x1 and x2
x1 = np.linspace(-1.5, 1.5, 100)
# decision boundary
x2 = -(x1 * o1[0, 0] + b1[0]) / o1[0, 1]
plt.plot(x1, x2,c='r')
for id in range(3):
    plt.scatter(Xi[id, 0], Xi[id, 1], marker='s',c='k')
plt.scatter(Xi[3, 0], Xi[3, 1],marker="^",c='b')
plt.title("ADALINE WITH {} EPOCHS OF TRAINING".format(epochs))
plt.xlabel("X1")
plt.ylabel("X2")
#PLOT VERTICAL LINE ALONG Y AXIS
M=np.size(x1)
a=np.zeros(M)
b=np.linspace(-1.5,1.5,M)
plt.plot(a,b,c='b')

a=np.linspace(-1.5,1.5,M)
b=np.zeros(M)
plt.plot(a,b,c='b')
plt.ylim((-1.5,1.5))

plt.show()

