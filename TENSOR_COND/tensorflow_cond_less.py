#written by jorge orlando miranda ñahui
import tensorflow as tf
#use of graph and session
tf.compat.v1.disable_eager_execution()
#use of the less method of tensorflow
t1=tf.constant(2,tf.float32,name="T1")
t2=tf.constant(0,tf.float32,name="T2")
print(t1,t2)
#use tf.cond(pred,f1(),f2())
#returns a f1 if pred is true else f2
c1=tf.cond(tf.less(t1,t2),lambda :tf.constant(0,tf.float32),lambda :tf.constant(1,tf.float32))
#create a Session using the context manager 
with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as se1:
    #run and print the result in the session
    print(se1.run(c1))


