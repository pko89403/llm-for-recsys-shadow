from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.utils import init_all_seeds


class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--data_dir", type=str, required=True, help="input file")
        parser.add_argument("--dataset", type=str, required=True, choices=["ml-100k", "amazon"], help="output_file")
        parser.add_argument("--n_neg_items", type=int, default=7, help="numbers of negative items")

    def run(self, data_dir: str, dataset: dir, n_neg_items: int):
        init_all_seeds(2024)
        if dataset == "ml-100k":
            print("ml-100k")
            pass
        elif dataset == 'amazon':
            print("amazon")
            pass
        else:
            raise NotImplementedError
        
if __name__ == "__main__":
    preprocess = PreprocessTask()
    preprocess.launch()