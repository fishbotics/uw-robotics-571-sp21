import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)

    def forward(self, data):
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        pass

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        pass

    def backward(self, previous_partial_gradient):
        # TODO
        self.backward_numba(self.data, previous_partial_gradient)
        return None
