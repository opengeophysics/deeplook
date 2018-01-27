"""
Classes defining linear and non-linear models.

Implements common methods for inversion classes, such as model evaluation
metrics.
"""


class LinearModel():

    def __init__(self, nparams):
        self.nparams = nparams
        self.params_ = None

    def r2_score(self, data, args):
        """
        Calculate the R2 coefficient of determination for the model.
        """
        predicted = self.predict(*args)
        total_sum_squares = ((data - data.mean())**2).sum()
        residual_sum_squares = ((data - predicted)**2).sum()
        return 1 - residual_sum_squares/total_sum_squares
