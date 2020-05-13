import cv2
import numpy as np

from kaggle_runner.utils import kernel_utils


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


class PS_result_analyzer:  # todo maybe add other good analyze functions
    def dev_set_performance(self, y_true, y_pred):
        return kernel_utils.dice_coef(y_true, y_pred)