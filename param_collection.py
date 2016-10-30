import tensorflow as tf
import numpy as np

class ParamCollection(object):

    def __init__(self, sess, params):
        """
        params should be a list of TensorFlow nodes.
        """
        self._params = params
        # Have to import the session to get the values being used.
        self.sess = sess
        self.sess.run(tf.initialize_variables(params))

    @property
    def params(self):
        return self._params

    def get_values(self):
        """
        Returns list of values of parameter arrays
        """
        return [self.sess.run(param) for param in self._params]

    def get_shapes(self):
        """
        Shapes of parameter arrays
        """
        return [param.get_shape().as_list() for param in self._params]

    def get_total_size(self):
        """
        Total number of parameters
        """
        return sum(np.prod(shape) for shape in self.get_shapes())

    def num_vars(self):
        """
        Number of parameter arrays
        """
        return len(self._params)

    def set_values(self, parvals):
        """
        Set values of parameter arrays given list of values `parvals`
        """
        assert len(parvals) == len(self._params)
        for (param, newval) in zip(self._params, parvals):
            self.sess.run(tf.assign(param, newval))
            assert tuple(param.get_shape().as_list()) == newval.shape

    def set_values_flat(self, theta):
        """
        Set parameters using a vector which represents all of the parameters
        flattened and concatenated.
        """
        arrs = []
        n = 0
        for shape in self.get_shapes():
            size = np.prod(shape)
            arrs.append(theta[n:n+size].reshape(shape))
            n += size
        assert theta.size == n
        self.set_values(arrs)

    def get_values_flat(self):
        """
        Flatten all parameter arrays into one vector and return it as a numpy array.
        """
        theta = np.empty(self.get_total_size())
        n = 0
        for param in self._params:
            s = np.prod(param.get_shape().as_list())
            theta[n:n+s] = self.sess.run(param).flatten()
            n += s
        assert theta.size == n
        return theta

    def _params_names(self):
        return [(param, param.name) for param in self._params]

    def to_h5(self, grp):
        """
        Save parameter arrays to hdf5 group `grp`
        """
        for (param, name) in self._params_names():
            arr = self.sess.run(param)
            grp[name] = arr

    def from_h5(self, grp):
        parvals = [grp[name] for(_, name) in self._params_names()]
        self.set_values(parvals)
