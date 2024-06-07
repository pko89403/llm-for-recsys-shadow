import os
import sys
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks import PreprocessTask
from macrec.tasks import EvaluateTask

def main():
    init_parser = ArgumentParser()
    init_parser.add_argument("-m", "--main", type=str, required=True, help="The main function to run")
    init_parser.add_argument("--verbose", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="The log level")
    init_args, init_extras = init_parser.parse_known_args() # 미리 정해진 인자 뿐만 아니라 추가로 인자를 받고 싶을 때 사용한다

    logger.remove()
    logger.add(sys.stderr, level=init_args.verbose)
    os.makedirs('logs', exist_ok=True)
    # log name을 프로그램 시작 시간으로 설정, 로그 레벨 INFO
    logger.add("logs/{time:YYYY-MM-DD:HH:mm:ss}.log", level="INFO")

    try:
        task = eval(init_args.main + "Task")()
    except NameError:
        logger.error(f"Task {init_args.main} not found")
        return
    task.launch()

if __name__ == "__main__":
    main()

