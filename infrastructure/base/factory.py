from abc import abstractmethod, ABC

class Factory(ABC):

    @abstractmethod
    def create(self):
        pass