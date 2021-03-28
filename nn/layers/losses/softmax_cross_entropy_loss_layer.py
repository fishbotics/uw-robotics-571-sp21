import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        return 0

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        return None
