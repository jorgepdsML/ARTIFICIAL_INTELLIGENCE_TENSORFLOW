#written by jorge orlando miranda ñahui
#visit and suscribe to my youtube's channel : jorge miranda redes neuronales
#github repo pdsjorgeML
#perceptron learning rule
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
#execute the graph using session
tf.compat.v1.disable_eager_execution()
#reset the default graph
tf.compat.v1.reset_default_graph()
#----------TRAINING AND TESTING DATA------------------------------------
Xi=np.array([[-1,-1],[-1,1],[1,1.0],[1.0,-1]],dtype=float)
Ti=np.array([-1.0,-1.0,1.0,-1.0],dtype=float)
#----------Define an empty graph with tf.Graph()
g=tf.Graph()
#DEFINITIONS
#SINGLE LAYER
R=2 # NUMBER OF INPUTS
S=1 # NUMBER OF NEURONES
#define operations inside the graph
with g.as_default():
    #placeholders for passing values later(training data points)
    x=tf.compat.v1.placeholder(tf.float64,shape=(R,1),name="X")
    T=tf.compat.v1.placeholder(tf.float64,shape=(S,1),name="T")
    #Variables to define the weights and bias
    w=tf.Variable(np.zeros([S,R]),tf.float64,name="W")
    #w=tf.Variable(np.random.randn(S,R),tf.float64,name="W")
    b=tf.Variable(np.zeros([S,1]),tf.float64,name="b")
    #b=tf.Variable(np.random.randn(S,1),tf.float64,name="b")
    #dot product between X and W
    z=tf.matmul(w,x)
    # add operation to define the net input of the activation function
    n = tf.add(z, b, name="ADICION")
    # --------------activation function----------
    # use of hardlims activation function
    y = tf.sign(n)
    # define the error of desired response and actual response
    E = tf.subtract(T, y, name="RESTA")
    # matmul
    delta = tf.matmul(E, tf.transpose(x))
    # algorithm of perceptron
    lr = 0.5  # learning rate

    w_up = tf.compat.v1.assign(w, w + delta, name="rule")
    b_up = tf.compat.v1.assign(b, b + E, name="bias")
# ------CREATE THE SESSION TO RUN THE  COMPUTATIONAL GRAPH
with tf.compat.v1.Session(graph=g) as se1:
    # initialize global variables
    se1.run(tf.compat.v1.global_variables_initializer())
    # epochs
    epochs = 10
    for n_epoch in range(epochs):
        # number of incorrect classification
        cont = 0
        # iterate over all the examples
        for id in range(4):
            loss = se1.run(E, feed_dict={x: np.reshape(Xi[id, :], (2, 1)), T: np.reshape(Ti[id], (1, 1))})
            o1 = se1.run(w_up, feed_dict={x: np.reshape(Xi[id, :], (2, 1)), T: np.reshape(Ti[id], (1, 1))})
            b1 = se1.run(b_up, feed_dict={x: np.reshape(Xi[id, :], (2, 1)), T: np.reshape(Ti[id], (1, 1))})
            # add 1 if there is an error of classification
            if int(loss) != 0:
                cont = cont + 1
        # if all the classifications are ok , break the inner for loop
        if cont == 0:
            print("CORRECT CLASSIFICATION")
            print("N° OF EPOCH : ", n_epoch)
            break
        cont = 0
    # generate the event file to visualize with tensorboard
    #writer = tf.compat.v1.summary.FileWriter('grafo_perceptron', se1.graph)
# TESTING THE WEIGHTS AND BIAS VALUES

from matplotlib import pyplot as plt

# define x1 and x2
x1 = np.linspace(-1, 1, 100)
# decision boundary
x2 = -(x1 * o1[0, 0] + b1[0]) / o1[0, 1]
plt.plot(x1, x2)
for id in range(3):
    plt.scatter(Xi[id, 0], Xi[id, 1], c='b')
plt.scatter(Xi[3, 0], Xi[3, 1], c='r')
plt.show()
