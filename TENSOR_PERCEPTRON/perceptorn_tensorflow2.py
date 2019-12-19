#written by jorge orlando miranda ñahui
#this is a project to classify 2 classes based on a OR logic operation
#perceptron architecture
#github repo pdsjorgeML
#perceptron learning rule
#import necessary tools
import numpy as np
import tensorflow as tf
#execute the graph using session
tf.compat.v1.disable_eager_execution()
#reset the default graph
tf.compat.v1.reset_default_graph()
#----------TRAINING AND TESTING DATA------------------------------------
Xd=np.array([[0,0],[0,1],[1,0],[1.0,1.0]],dtype=float)
Td=np.array([0,0,0,1.0],dtype=float) #CLASS 0  , CLASS 1
#Number of training data points
N=np.size(Td)
#handle the default graph
grafo=tf.compat.v1.get_default_graph()
#use of the context manager with the block with
with grafo.as_default():
    # add operations to the default graph
    #R number of inputs  , S number of neurones of the single network
    R,S=2,1
    with tf.name_scope("DATA"):
        #definition of placeholders Xi and Ti
        Xi=tf.compat.v1.placeholder(tf.float64,shape=(R,1),name="Xi_training")
        Ti=tf.compat.v1.placeholder(tf.float64,shape=(S,1),name="Ti_training")

    # Variables to define the weights and bias
    w = tf.Variable(np.random.randn(S, R), tf.float64, name="Weights")
    # w=tf.Variable(np.random.randn(S,R),tf.float64,name="W")
    b = tf.Variable(np.random.randn(S, 1), tf.float64, name="Bias")
    #net input
    z=tf.matmul(w,Xi,name="MatMul_operation")
    #add method
    v=tf.add(z,b,name="Net_Input")
    with tf.name_scope("Heaviside"):
        #hardlim function
        y=tf.nn.relu(tf.sign(v, name="Heaviside_Function"))
    with tf.name_scope("ERROR"):
        #ERROR
        E=tf.subtract(Ti,y,name="Subtract_method")
    #LEARNING RATE PARAMETER OF PERCEPTRON LEARNING RULE
    lr=0.5
    with tf.name_scope("W_UPDATE"):
        W_up=tf.compat.v1.assign(w,w+lr*tf.matmul(E,tf.transpose(Xi)),name="Weights_update")
    with tf.name_scope("B_UPDATE"):
        B_up=tf.compat.v1.assign(b,b+lr*E*1,name="biasupdate")
with tf.compat.v1.Session(graph=grafo) as sesion1:
    sesion1.run(tf.compat.v1.global_variables_initializer())
    #number of epochs , adjuts this value
    epocas=20
    #count the times the error of each training point is 0 or different
    cont=0
    #training of the single neural network (perceptron architecture)
    for epoca in range(epocas):
        flag=0
        for ejemplo in range(4):
            # run the error
            ech = sesion1.run(E,feed_dict={Xi: np.reshape(Xd[ejemplo, :], (2, 1)), Ti: np.reshape(Td[ejemplo], (S, 1))})
            ww = sesion1.run(w,feed_dict={Xi: np.reshape(Xd[ejemplo, :], (2, 1)), Ti: np.reshape(Td[ejemplo], (S, 1))})
            bb = sesion1.run(b,feed_dict={Xi: np.reshape(Xd[ejemplo, :], (2, 1)), Ti: np.reshape(Td[ejemplo], (S, 1))})
            if ech == 0:
                cont = cont + 1
                if cont==N:
                    flag=1
                    break
            wch = sesion1.run(W_up,feed_dict={Xi: np.reshape(Xd[ejemplo, :], (2, 1)), Ti: np.reshape(Td[ejemplo], (S, 1))})
            bch = sesion1.run(B_up,feed_dict={Xi: np.reshape(Xd[ejemplo, :], (2, 1)), Ti: np.reshape(Td[ejemplo], (S, 1))})
        #determine if all the training data points are correctly classified
        if flag==1:
            print("LA RED NEURONAL APRENDIO CORRECTAMENTE")
            #print the values to the shell
            print(wch,bch)
            print("NÚMERO DE EPOCA",epoca+1)
            #CREATE EVENT FILES IN your_directory
            #tf.compat.v1.summary.FileWrite("your_directory",session1.your_graph)
            break
        else:
            cont=0
    #FileWriter summary
    writer = tf.compat.v1.summary.FileWriter("logdir_graph", sesion1.graph)
writer.close()
#PLOT
if flag==1:
    # visualize the training samples and the decision plane
    from matplotlib import pyplot as plt

    # for class 1
    for idx in range(N - 1):
        plt.scatter(Xd[idx, 0], Xd[idx, 1], color='green')
    # for class 2
    plt.scatter(Xd[N - 1, 0], Xd[N - 1, 0], color='red')
    # decision surface
    w1 = np.linspace(0, 1.2, 1000)
    w2 = (-bch[0] - w1 * wch[0, 0]) / wch[0, 1]
    # -------------
    # plot the decision surface(place))
    # with the weights and bias
    plt.plot(w1, w2, color='blue')
    plt.show()
