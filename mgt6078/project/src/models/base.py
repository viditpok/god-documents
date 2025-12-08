"""
base interface for peer regressors
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.utils.logging import get_logger


class PeerRegressor(ABC):
    """
    abstract peer regressor replication scaffold
    """

    # guideline §5 empirical strategy l37-l52

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def fit(self, X, y) -> "PeerRegressor":
        """
        train model on feature frame and target vector
        """

    @abstractmethod
    def predict(self, X):
        """
        infer peer-implied fair value predictions
        """
