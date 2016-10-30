#! /usr/bin/env python
# -*- coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
from pprint import pprint
from param_collection import ParamCollection

# Helper functions.
def weight_variable(shape, stddev=0.1, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape, init_bias=0.1, initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

class InitialColumnProgNN(object):
    """
    Descr: Initial network to train for later use transfer learning with a
        Progressive Neural Network.
    Args:
        topology - A list of number of units in each hidden dimension.
                   First entry is input dimension.
        activations - A list of activation functions to use on the transforms.
        session - A TensorFlow session.
    Returns:
        None - attaches objects to class for InitialSingleColumn.session.run()
    """

    def __init__(self, topology, activations, session, dtype=tf.float64):
        n_input = topology[0]
        # Layers in network.
        L = len(topology) - 1
        self.session = session
        self.L = L
        self.topology = topology
        self.o_n = tf.placeholder(dtype,shape=[None, n_input])

        self.W = []
        self.b =[]
        self.h = [self.o_n]
        params = []
        for k in range(L):
            shape = topology[k:k+2]
            self.W.append(weight_variable(shape))
            self.b.append(bias_variable([shape[1]]))
            self.h.append(activations[k](tf.matmul(self.h[-1], self.W[k]) + self.b[k]))
            params.append(self.W[-1])
            params.append(self.b[-1])
        self.h = self.h
        self.pc = ParamCollection(self.session, params)


class ExtensibleColumnProgNN(object):
    """
    Descr: An extensible network column for use in transfer learning with a
        Progressive Neural Network.
    Args:
        topology - A list of number of units in each hidden dimension.
            First entry is input dimension.
        activations - A list of activation functions to use on the transforms.
        session - A TensorFlow session.
        prev_columns - Previously trained columns, either Initial or Extensible,
            we are going to create lateral connections to for the current column.
    Returns:
        None - attaches objects to class for InitialSingleColumn.session.run()
    """

    def __init__(self, topology, activations, session, prev_columns, dtype=tf.float64):
        n_input = topology[0]
        self.topology = topology
        self.session = session
        width = len(prev_columns)
        # Layers in network. First value is n_input, so it doesn't count.
        L = len(topology) -1
        self.L = L
        self.prev_columns = prev_columns

        # Doesn't work if the columns aren't the same height.
        assert all([self.L == x.L for x in prev_columns])

        self.o_n = tf.placeholder(dtype, shape=[None, n_input])

        self.W = [[]] * L
        self.b = [[]] * L
        self.U = []
        for k in range(L-1):
            self.U.append( [[]] * width )
        self.h = [self.o_n]
        # Collect parameters to hand off to ParamCollection.
        params = []
        for k in range(L):
            W_shape = topology[k:k+2]
            self.W[k] = weight_variable(W_shape)
            self.b[k] = bias_variable([W_shape[1]])
            if k == 0:
                self.h.append(activations[k](tf.matmul(self.h[-1],self.W[k]) + self.b[k]))
                params.append(self.W[k])
                params.append(self.b[k])
                continue
            preactivation = tf.matmul(self.h[-1],self.W[k]) + self.b[k]
            for kk in range(width):
                U_shape = [prev_columns[kk].topology[k], topology[k+1]]
                self.U[k-1][kk] = weight_variable(U_shape)
                # pprint(prev_columns[kk].h[k].get_shape().as_list())
                # pprint(self.U[k-1][kk].get_shape().as_list())
                preactivation +=  tf.matmul(prev_columns[kk].h[k],self.U[k-1][kk])
            self.h.append(activations[k](preactivation))
            params.append(self.W[k])
            params.append(self.b[k])
            for kk in range(width):
                params.append(self.U[k-1][kk])

        self.pc = ParamCollection(self.session, params)


def test_ProgNN():
    # Make some fake observations.
    fake1 = np.float64(np.random.rand(4000,128))
    fake2 = np.float64(np.random.rand(4000,128))
    fake3 = np.float64(np.random.rand(4000,128))
    fake4 = np.float64(np.random.rand(4000,128))
    fake5 = np.float64(np.random.rand(4000,128))
    n_input = 128
    topology1 = [n_input, 100, 64, 25, 9]
    topology2 = [n_input, 68, 44, 19, 7]
    topology3 = [n_input, 79, 58, 33, 12]
    topology4 = [n_input, 40, 30, 20, 10]
    topology5 = [n_input, 101, 73, 51, 8]
    activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax]

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    col_0 = InitialColumnProgNN(topology1, activations, session)
    th0 = col_0.pc.get_values_flat()
    col_1 = ExtensibleColumnProgNN(topology2, activations, session, [col_0])
    th1 = col_1.pc.get_values_flat()
    col_2 = ExtensibleColumnProgNN(topology3, activations, session, [col_0, col_1])
    th2 = col_2.pc.get_values_flat()
    col_3 = ExtensibleColumnProgNN(topology4, activations, session, [col_0, col_1, col_2])
    th3 = col_3.pc.get_values_flat()
    col_4 = ExtensibleColumnProgNN(topology5, activations, session, [col_0, col_1, col_2, col_3])
    th4 = col_4.pc.get_values_flat()

    # This pattern to evaluate the Progressive NN can be extended to a
    # arbitrarily large number of columns / models.

    # Fake train the first network. h_0[-1] has information loss functions need.
    h_0 = col_0.session.run([col_0.h],
        feed_dict={col_0.o_n:fake1})

    # Fake train the second network, but this time with lateral connections to
    # fake pre-trained, constant weights from first column of Progressive NN.
    h_1 = col_1.session.run([col_1.h],
        feed_dict={col_1.o_n:fake2, col_1.prev_columns[0].o_n:fake2})

    # Now fake train a third column that has lateral connections to both
    # previously "trained" columns.
    h_2 = col_2.session.run([col_2.h],
        feed_dict={col_2.o_n:fake3,
            col_2.prev_columns[0].o_n:fake3,
            col_2.prev_columns[1].o_n:fake3})

    # Fourth column / fake instance of training.
    h_3 = col_3.session.run([col_3.h],
        feed_dict={col_3.o_n:fake4,
            col_3.prev_columns[0].o_n:fake4,
            col_3.prev_columns[1].o_n:fake4,
            col_3.prev_columns[2].o_n:fake4})

    # Fifth column. Notice we have to pass in n placeholder with the same
    # obsevations to a Progressive NN with n columns.
    h_4 = col_4.session.run([col_4.h],
        feed_dict={col_4.o_n:fake5,
            col_4.prev_columns[0].o_n:fake5,
            col_4.prev_columns[1].o_n:fake5,
            col_4.prev_columns[2].o_n:fake5,
            col_4.prev_columns[3].o_n:fake5})

    # Anyway, you get the drift. Hope this helps someone understand
    # Progressive Neural Networks!

    # Make sure the column parameters aren't changing when being used by
    # later columns.
    
    # Should be a list of [0., 0., 0., ... 0.] if theta isn't changing.
    # We add 1.0 to each element to see if they were all zero with np.all().
    assert np.all(col_4.prev_columns[0].pc.get_values_flat() - th0 + 1.)

if __name__ == "__main__":
    test_ProgNN()
