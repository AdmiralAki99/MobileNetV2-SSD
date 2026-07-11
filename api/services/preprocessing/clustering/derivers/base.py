from abc import ABC, abstractmethod
from ..results import ClusterResult

class PriorsDeriver(ABC):
    @abstractmethod
    def derive_priors(self, box_dimenstions, strategy, params):
        pass