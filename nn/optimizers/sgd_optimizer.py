from .base_optimizer import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate):
        super(SGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            # TODO fix the line below to apply the parameter update
            parameter.data = None
