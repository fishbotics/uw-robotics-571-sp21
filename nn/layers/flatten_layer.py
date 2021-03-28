from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None
