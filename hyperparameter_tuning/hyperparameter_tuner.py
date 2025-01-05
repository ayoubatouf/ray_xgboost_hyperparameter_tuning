from abc import ABC, abstractmethod


class HyperparameterTuner(ABC):
    @abstractmethod
    def tune(self):
        pass
