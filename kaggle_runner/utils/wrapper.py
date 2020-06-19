from functools import partial, wraps

from kaggle_runner import may_debug


def size_decorator(f):
    @wraps(f)
    def size_wrapper(y_true, y_pred, *args, **kwargs):
        ts = y_true.shape
        ps = y_pred.shape

        if len(ps) > len(ts) or (
                (ps[1] > ts[1]) if ts[1] is not None else False):
            y_pred = y_pred[:,0]

        return f(y_true, y_pred, *args, **kwargs)

    return size_wrapper
