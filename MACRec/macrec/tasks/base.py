from abc import ABC, abstractmethod
from argparse import ArgumentParser
from loguru import logger
from typing import Any


class Task(ABC):
    @staticmethod
    @abstractmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        """_summary_

        Args:
            parser (ArgumentParser): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            ArgumentParser: _description_
        """
        raise NotImplementedError
    
    def __getattr__(self, __name: str) -> Any:
        # return none if attribute is not found
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"Task {self.__class__.__name__} has no attribute {__name}")
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """_summary_

        Args:
            *args: _description_
            **kwargs: _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def launch(self) -> Any:

        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        self.args = args
        # log the arguments
        logger.success(args)
        return self.run(**vars(args))