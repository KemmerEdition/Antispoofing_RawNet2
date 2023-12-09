from hw_5.metric.EER_utils import compute_eer
import numpy as np


class EERMetric:
    def __init__(self):
        self.name = "EERMetric"

    def __call__(self, target: np.array, predict: np.array, **kwargs):
        eer_value = compute_eer(predict[target == 1], predict[target == 0])[0]
        return eer_value
