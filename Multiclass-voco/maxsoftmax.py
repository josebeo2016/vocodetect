"""
https://github.com/kkirchheim/pytorch-ood/blob/dev/src/pytorch_ood/detector/softmax.py
"""
from typing import Optional, TypeVar

from torch import Tensor
from torch.nn import Module

Self = TypeVar("Self")
class MaxSoftmax():
    """
    Implements the Maximum Softmax Probability (MSP) Thresholding baseline for OOD detection.

    Optionally, implements temperature scaling, which divides the logits by a constant temperature :math:`T`
    before calculating the softmax.

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function and :math:`\\sigma_y`  indicates the :math:`y^{th}` value of the
    resulting probability vector.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/1610.02136>`_
    :see Implementation:
        `GitHub <https://github.com/hendrycks/error-detection>`_

    """

    def __init__(self, model: Module, t: Optional[float] = 1.0):
        """
        :param model: neural network to use
        :param t: temperature value :math:`T`. Default is 1.
        """

        self.t = t
        self.model = model
        
    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through model
        :return: maximum softmax probability and its index
        """
        if self.model is None:
            raise ModelNotSetException
        model_out = self.model(x)
        max_idx = model_out.softmax(dim=1).max(dim=1).indices
        return self.score(model_out, t=self.t), max_idx

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model
        """
        return MaxSoftmax.score(logits, self.t)

    @staticmethod
    def score(logits: Tensor, t: Optional[float] = 1.0) -> Tensor:
        """
        :param logits: logits for samples
        :param t: temperature value
        """
        return -logits.div(t).softmax(dim=1).max(dim=1).values