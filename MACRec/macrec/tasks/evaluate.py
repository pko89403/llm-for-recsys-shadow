import os 
import jsonlines
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.generation import GenerationTask
from macrec.utils import str2list, NumpyEncoder
from macrec.evaluation import 


class EvaluateTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument("--steps", type=int, default=2, help="Number of steps")
        parser.add_argument("--topks", type=str2list, default=[1, 3, 5], help="Top-Ks for ranking task")
        return parser
    
    def get_metrics(self, topks: list[int] = [1, 3, 5]):
        if self.task == "rp":
            self.metrics = MetricDict({

            })