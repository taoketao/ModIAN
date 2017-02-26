"""
2/25/17   Morgan Bryant

Arguments/Parameters: Invoke as 'python ModIAN.py -h' or 'ipython ModIAN.py -- -h'

See info.txt for information.  
This was tensorflow implementation was adapted from my mram.py, an mult-attention
extension of recurrent-attention-model.

This is a basic implementation of a psychologically-grounded neural network model 
that uses a simple selection scheme that extends the model's ability to perform
significant transferrence of aggregated skill.

[ see associated documents for details. ]

"""

#########################   IMPORTS & VERSIONS & SETUP #########################
import tensorflow as tf
#import tf_mnist_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time, random, sys, os, argparse
from datetime import datetime
startTime = datetime.now()

inp_sz = 5;     lr = 0.001
N = 10;         epochs = 10
theta = 8;      batchsize = 16
B = 4;          disp_step = 10
phi = 16
out_sz = 2

def to_matrices(D):
    n = len(D)
    r1 = len(D[0]['x']); r2 = len(D[0]['y'])
    print n, r1, r2
    X = np.zeros( (n,r1) )
    Y = np.zeros( (n,r2) )
    print X.shape, Y.shape
    for di,d in enumerate(D):
        for xi in range(r1):
            X[di,xi] = d['x'][xi]
        for yi in range(r2):
            Y[di,yi] = d['y'][yi]
    return X,Y

def simple_task(n=100, I=inp_sz, O=out_sz):
    l = []
    for i in range(n):
        x = tuple([random.randint(-5,20) for _ in range(I)])
        y = [x[0]+x[1], x[2]-x[3]-sum(x[4:]) ]
        while len(y)<O:
            y = y+[0]
        y=tuple(y)
        l.append( {'x':x, 'y':y} )
    return to_matrices(l)

# DATA
train_I, train_I = simple_task(200)
test_I, test_I = simple_task(30)

# I -> X_t.  I -> X_B.  sigma( (MX_t)sigma(X_B) ) -> Y.  Y -> X'.
#   inp_sz = 5;     lr = 0.001
#   N = 10;         epochs = 10
#   theta = 8;      batchsize = 16
#   B = 4;          disp_step = 10
#   phi = 16
#   out_sz = 2

def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval = -0.1, maxval = 0.1, dtype='float32')
    return tf.Variable(initial, name=myname, trainable=train)

def model(inp, out, w):
  with tf.variable_scope("network", reuse=None):
#    X_B = tf.add(tf.matmul(x, w['W_I_XB']), w['b_I_XB'])
#    X_th = tf.add(tf.matmul(x, w['W_I_Xth']), w['b_I_Xth'])

    print '------------'
    print inp.get_shape(), w['W_I_Xth'].get_shape(), w['W_I_XB'].get_shape()
    inp_r = tf.reshape(inp, (inp_sz, 1))
    W_I_XB_r = tf.reshape(w['W_I_XB'], (N*B, inp_sz))
    W_I_Xth_r = tf.reshape(w['W_I_Xth'], (N*theta, inp_sz))
    X_B  = tf.matmul(W_I_XB_r, inp_r )
    X_th = tf.matmul(W_I_Xth_r, inp_r )
    X_B = tf.nn.softmax(X_B)


    X_B = tf.reshape(X_B, (N, B))
    X_th = tf.reshape(X_th, (N, theta))
    # Todo: using mapfn, apply M to each of N: to X_th then to X_B.
    # After the map, combine into Y and [FC] to X'.

    def working_test_of__map_fn():
        M_th_B = tf.reshape(w['M'], (B * phi, theta))
        X_sl = tf.reshape(tf.slice(X_th, [0,0], [1,-1]), (theta,1))
        print M_th_B.get_shape(), X_th.get_shape(), X_sl.get_shape()
        Z = tf.matmul(M_th_B, X_sl)
        Zr= tf.reshape(Z, (phi, B))
        X_B = tf.reshape(X_B, (B,N))
        print Zr.get_shape() ,'\n', X_B.get_shape()
        def TEMP_module_operation_ordered_TH_B( input_xb):
            Y = tf.matmul(Zr, X_B)
            return Y

        Y = tf.map_fn(TEMP_module_operation_ordered_TH_B, X_B)
        print Y.get_shape() ## YES!
        sys.exit()

    D = 'float32'
    x_th = tf.reshape(X_th, (N, theta, 1))
    x_b = tf.reshape(X_B, (N, B, 1))

    M_th_B = tf.reshape(w['M'], (B * phi, theta))
    def module_operation_TH(x_th):
        Z = tf.matmul(M_th_B, x_th) # shape: (B*phi, N). -> (N*B, phi)
        return Z
    def module_operation_B(i):
        Z_slice = tf.reshape(tf.slice(Zr, [i,0,0],[1,-1,-1]), (phi,B))
        xb_slice = tf.reshape(tf.slice(x_b, [i,0,0],[1,-1,-1]), (B,1))
        return tf.matmul(Z_slice, xb_slice)

    Z = tf.map_fn(module_operation_TH, x_th, dtype=D)
    Zr = tf.reshape(Z, (N,  phi, B,))
    print "T", tf.slice(Zr, [3,0,0],[1,-1,-1]).dtype
    Y = tf.map_fn(module_operation_B, tf.range(N), dtype=D) # map_fn: for parallelization

    Yr = tf.reshape(Y, (N*phi,1))
    FC_Y_Xnew = tf.reshape(w['W_Y_X1'], (out_sz, N*phi))
    X_new = tf.nn.relu( tf.matmul(FC_Y_Xnew, Yr) )
    return tf.reshape(X_new, (out_sz,))


with tf.Graph().as_default():

    Weights_0 = {
      'W_I_Xth': weight_variable((inp_sz, N, theta), "Weights_Inp_to_Xtheta",True),
      'b_I_Xth': weight_variable((1, N, theta), "biases_Inp_to_Xtheta", True),
      'W_I_XB' : weight_variable((inp_sz, N, B), "Weights_Inp_to_XB", True),
      'b_I_XB' : weight_variable((1, N, B), "biases_Inp_to_XB", True),
      'M'      : weight_variable( (B, theta, phi), "Module", True), # trainable
        #M = weight_variable( (theta, 1, 1), "Module_biases", True) # trainable
      'W_Y_X1' : weight_variable( (N, phi, out_sz), "Weights_Y_to_Xprime", True),
      'b_Y_X1' : weight_variable( (N, 1, out_sz), "biases_Y_to_Xprime", True),
    }
    print '###############'
    for k,v in Weights_0.items():
        print k, ':', v.get_shape()
    print '###############'
    y_var = tf.placeholder("float32", [out_sz])
    x_var = tf.placeholder("float32", [inp_sz])
    print x_var.get_shape(), y_var.get_shape()
    this_model = model(x_var, y_var, Weights_0)




print "Done."
