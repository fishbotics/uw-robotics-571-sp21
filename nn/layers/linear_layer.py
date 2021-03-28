from typing import Optional, Callable

import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(LinearLayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))
        self.weight = None  # TODO create the weight parameter
        self.initialize()

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Linear conv_layers (fully connected) forward pass
        :param data: n X d array (batch x features)
        :return: n X c array (batch x channels)
        """
        # TODO do the linear conv_layers
        return None

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future conv_layers
        :return: gradients wrt inputs
        """
        # TODO do the backward step
        return None

    def selfstr(self):
        return str(self.weight.data.shape)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(LinearLayer, self).initialize()
