import numpy as np
from recbole.evaluator import AbstractMetric, DataStruct
from recbole.utils import EvaluatorType


class PopularityEntropy(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject: DataStruct):
        """Get the matrix of recommendation items and the popularity of items in training data"""
        item_counter = dataobject.get("data.count_items")
        item_matrix = dataobject.get("rec.items")
        return item_matrix.numpy(), dict(item_counter)

    def calculate_metric(self, dataobject):
        item_matrix, item_count = self.used_info(dataobject)
        result = self.metric_info(self.get_pop(item_matrix, item_count))
        metric_dict = self.topk_result("popularityentropy", result)
        return metric_dict

    def get_pop(self, item_matrix, item_count):
        """Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            item_count(dict): the number of interaction of items in training data.

        Returns:
            numpy.ndarray: the popularity of items in the recommended list.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict
