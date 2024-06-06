import torchmetrics
from loguru import logger

class MetricDict:
    def __init__(self, metrics: dict[str, torchmetrics.Metric] = {}):
        self.metrics: dict[str, torchmetrics.Metric] = metrics

    def add(self, name: str, metric: torchmetrics.Metric):
        self.metrics[name] = metric

    def update(self, output: dict, prefix: str = "") -> str:
            """주어진 출력(output)을 사용하여 메트릭을 업데이트합니다. 주어진 접두사(prefix)로 시작하는 메트릭을 업데이트합니다.

            Args:
                output (dict): 업데이트에 사용할 출력입니다.
                prefix (str, optional): 메트릭 이름의 접두사입니다. 기본값은 ""입니다.

            Returns:
                str: 업데이트된 메트릭의 결과를 나타내는 문자열입니다.
            """
            for metric_name, metric in self.metrics.items():
                if not metric_name.startswith(prefix):
                    continue
                metric.update(output)
                computed = metric.compute()
                if len(computed) == 1:
                    # 첫 번째 아이템을 가져옵니다.
                    computed = next(iter(computed.values()))
                    logger.debug(f"{metric_name}: {computed:.4f}")
                else:
                    # 소수점 이하 4자리까지 모든 메트릭을 출력합니다.
                    logger.debug(f"{metric_name}:")
                    for key, value in computed.items():
                        logger.debug(f"{key}: {value:.4f}")
            # 첫 번째 메트릭과 해당 메트릭의 이름을 가져옵니다.
            metric_name, metric = next(iter(self.metrics.items()))
            if not metric_name.startswith(prefix):
                return ''
            computed = metric.compute()
            computed = next(iter(computed.values()))
            return f"{metric_name}: {computed:.4f}"

    def compute(self):
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric.compute()
        return result
    
    def report(self):
        result = self.compute()
        for metric_name, metric in result.items():
            if len(metric) == 1:
                # get first item
                metric = next(iter(metric.values()))
                logger.success(f"{metric_name}: {metric:.4f}")
            else:
                # output every metric with 4 decimal places
                logger.success(f"{metric_name}:")
                for key, value in metric.items():
                    logger.success(f"{key}: {value:.4f}")