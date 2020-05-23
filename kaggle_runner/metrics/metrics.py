import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import torch

from kaggle_runner import may_debug


def matthews_correlation_aux_stripper(y_true, y_pred):
    ts = y_true.shape
    ps = y_pred.shape

    if len(ps) > len(ts) or (
            (ps[1] > ts[1]) if ts[1] is not None else False):
        y_pred = y_pred[:,0]

    return matthews_correlation(y_true, y_pred)

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def metric(probability, truth, threshold=0.5, reduction="none"):
    """Calculates dice of positive and negative images seperately"""
    """probability and truth must be torch tensors"""
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def binary_sensitivity_np(y_pred, y_true):
    threshold = 0.5
    # predict_false = y_pred <= threshold
    y_true = y_true > threshold
    predict_true = y_pred > threshold
    TP = np.multiply(y_true, predict_true)
    # FP = np.logical_and(y_true == 0, predict_true)

    # as Keras Tensors
    TP = TP.sum()
    # FP = FP.sum()

    sensitivity = TP / y_true.sum()

    return sensitivity


def binary_sensitivity(y_pred, y_true):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """
    threshold = 0.5
    TP = np.logical_and(K.eval(y_true) == 1, K.eval(y_pred) <= threshold)
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) > threshold)

    # as Keras Tensors
    TP = K.sum(K.variable(TP))
    FP = K.sum(K.variable(FP))

    sensitivity = TP / (TP + FP + K.epsilon())

    return sensitivity


def binary_specificity(y_pred, y_true):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """

    threshold = 0.5
    TN = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) <= threshold)
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) > threshold)

    # as Keras Tensors
    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))

    specificity = TN / (TN + FP + K.epsilon())

    return specificity


def binary_auc_probability(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-12):
    """
    refer to this: https://blog.revolutionanalytics.com/2016/11/calculating-auc.html

    The probabilistic interpretation is that if you randomly choose a positive case and a negative case, the probability that the positive case outranks the negative case according to the classifier is given by the AUC. This is evident from the figure, where the total area of the plot is normalized to one, the cells of the matrix enumerate all possible combinations of positive and negative cases, and the fraction under the curve comprises the cells where the positive case outranks the negative one.

    :param y_true:
    :param y_pred:
    :param threshold:
    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    """

    # labels: y_true, scores: y_pred, N the size of sample
    # auc_probability < - function(labels, scores, N=1e7)
    # {
    #     pos < - sample(scores[labels], N, replace=TRUE)
    # neg < - sample(scores[!labels], N, replace = TRUE)
    # # sum( (1 + sign(pos - neg))/2)/N # does the same thing
    # (sum(pos > neg) + sum(pos == neg) / 2) / N  # give partial credit for ties
    # }

    # auc_probability(as.logical(category), prediction)

    threshold = math_ops.cast(threshold, y_pred.dtype)
    # y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

    true_pos_predict = math_ops.multiply(y_true, y_pred)  # %6 pos
    # 94% neg...
    true_neg_predict = math_ops.multiply(1.0 - y_true, 1 - y_pred)

    # recision = math_ops.div(correct_pos, predict_pos)
    # recall = math_ops.div(correct_pos, ground_pos)

    # if N_MORE:
    #    m = (2*recall*precision) / (precision+recall)
    # else:
    #    #m = (sensitivity + specificity)/2 # balanced accuracy
    #    raise NotImplementedError("Balanced accuracy metric is not implemented")

    # return m

def bin_prd_clsf_info_neg(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-7):
    """
    refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

    Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
     of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

    1) If N >> P, f1 is a better.

    2) If P >> N, b_acc is better.

    For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

    :param y_true:
    :param y_pred:
    :param threshold:
    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    """
    # if FOCAL_LOSS_GAMMA == 2.0:
    #    threshold = 0.57
    # elif FOCAL_LOSS_GAMMA == 1.0:
    #    threshold = (0.53 + (
    #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

    # ground_pos = math_ops.reduce_sum(y_true) + epsilon
    # correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
    # predict_pos = math_ops.reduce_sum(y_pred) + epsilon
    true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
    false_cnt = math_ops.reduce_sum(1 - y_true_b) + epsilon
    pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
    pred_false_cnt = math_ops.reduce_sum(1 - y_pred_b) + epsilon
    # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
    # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

    # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
    # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

    # tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
    # tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
    # precision = math_ops.div(correct_pos, predict_pos)
    # recall = math_ops.div(correct_pos, ground_pos)

    # if N_MORE:
    #    m = (2 * recall * precision) / (precision + recall)
    # else:
    #    # m = (sensitivity + specificity)/2 # balanced accuracy
    #    raise NotImplementedError("Balanced accuracy metric is not implemented")

    return (pred_false_cnt - false_cnt) / false_cnt  # (batchsize 1024)

def bin_prd_clsf_info_pos(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-7):
    """
    refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

    Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
     of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

    1) If N >> P, f1 is a better.

    2) If P >> N, b_acc is better.

    For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

    :param y_true:
    :param y_pred:
    :param threshold:
    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    """
    # if FOCAL_LOSS_GAMMA == 2.0:
    #    threshold = 0.57
    # elif FOCAL_LOSS_GAMMA == 1.0:
    #    threshold = (0.53 + (
    #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

    # ground_pos = math_ops.reduce_sum(y_true) + epsilon
    # correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
    # predict_pos = math_ops.reduce_sum(y_pred) + epsilon
    true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
    false_cnt = math_ops.reduce_sum(1 - y_true_b) + epsilon
    pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
    pred_false_cnt = math_ops.reduce_sum(1 - y_pred_b) + epsilon
    # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
    # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

    # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
    # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

    # tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
    # tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
    # precision = math_ops.div(correct_pos, predict_pos)
    # recall = math_ops.div(correct_pos, ground_pos)

    # if N_MORE:
    #    m = (2 * recall * precision) / (precision + recall)
    # else:
    #    # m = (sensitivity + specificity)/2 # balanced accuracy
    #    raise NotImplementedError("Balanced accuracy metric is not implemented")

    return (pred_true_cnt - true_cnt) / true_cnt  # (batchsize 1024)
