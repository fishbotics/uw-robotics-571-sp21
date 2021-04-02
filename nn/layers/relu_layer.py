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
