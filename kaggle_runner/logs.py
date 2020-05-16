import datetime
import time
from collections import defaultdict, deque

import numpy as np
import tensorflow as tf

import fastai
import torch
import torch.distributed as dist
from fastai.callbacks import csv_logger
from kaggle_runner.utils.kernel_utils import is_dist_avail_and_initialized


def metric_get_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(
        "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f"
        " | epoch: %d | phase: %s"
        % (epoch_loss, dice, dice_neg, dice_pos, iou, epoch, phase)
    )

    return dice, iou


class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(
            self,
            count_mode="samples",
            stateful_metrics=None,
            display_per_batches=1,
            verbose=1,
            early_stop=False,
            patience_displays=0,
            epsilon=1e-7,
    ):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = 1 if display_per_batches < 1 else display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.seen = 0
        self.verbose = verbose

        # better way is subclass EarlyStopping callback.
        self.early_stop = early_stop
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0
        self.batch_size = 0
        self.epochs = 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get("size", 0)
        # In case of distribution strategy we can potentially run multiple
        # steps at the same time, we should account for that in the `seen`
        # calculation.
        num_steps = logs.get("num_steps", 1)

        if self.use_steps:
            self.batch_size = num_steps
        else:
            self.batch_size = batch_size * num_steps

        before_seen = self.seen
        self.seen += self.batch_size
        after_seen = self.seen

        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.step_idx += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.

        if self.early_stop:
            # only record for this batch, not the display. Should work
            loss = logs.get("loss")
            self.losses_sum_display += loss

        if self.step_idx % self.display_per_batches == 0:
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)

            if self.early_stop:
                avg_loss_per_display = (
                    self.losses_sum_display / self.display_per_batches
                )
                self.losses_sum_display = 0  # clear mannually
                self.losses[
                    self.display_idx % self.patience_displays
                ] = avg_loss_per_display
                # but it still SGD, variance still, it just smaller by factor of
                # display_per_batches
                display_info_start_step = self.step_idx - self.display_per_batches + 1
                print(
                    f"\nmean(display): {avg_loss_per_display}, Step {display_info_start_step }({before_seen}) to {self.step_idx}({after_seen}) for {self.display_idx}th display step"
                )

                self.display_idx += 1  # used in index, so +1 later

                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses
                    )  # as SGD, always variance, so not a good way, need to learn from early stopping
                    print(
                        f"mean(over displays): {np.mean(self.losses)}, std:{std} for Display {self.display_idx-self.patience_displays} to {self.display_idx-1}"
                    )

                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f"Early Stop criterion met: std is {std} at Step"
                            f" {self.step_idx} for {self.display_idx}th display"
                            "steps"
                        )

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print("Step %05d: early stopping" % (self.stopped_step + 1))


class MetricLogger:
    def __init__(self, delimiter="\t", log_file_name="metric.log"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_file = open(log_file_name, "a", buffering=1)

    def __del__(self):
        self.log_file.close()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def clear(self):
        for meter in self.meters.values():
            if meter is not None:
                meter.clear()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]

        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []

        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))

        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 1

        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                # "max mem: {memory:.0f}",
            ]
        )
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            # print the first to let you know it is running....

            if i % (print_freq) == 0 or i == len(iterable):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.print_and_log_to_file(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        # memory=torch.cuda.max_memory_allocated() / MB, # FIXME add support for CPU
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.print_and_log_to_file(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

    def print_and_log_to_file(self, s):
        print(s)
        self.log_file.write(s + "\n")


class CSVLoggerBufferCustomized(csv_logger.CSVLogger):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."

    def __init__(
            self,
            learn: fastai.basic_train.Learner,
            filename: str = "history",
            append: bool = False,
            buffer_type: int = 1,
    ):
        super(CSVLoggerBufferCustomized, self).__init__(
            learn, filename, append)
        self.buffer_type = buffer_type  # flush the file to get quick result

    def on_train_begin(self, **kwargs) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = (
            self.path.open("a", buffering=self.buffer_type)

            if self.append
            else self.path.open("w", buffering=self.buffer_type)
        )
        self.file.write(
            ",".join(self.learn.recorder.names[: (
                None if self.add_time else -1)])
            + "\n"
        )


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def clear(self):
        self.total = 0.0
        self.count = 0
        self.deque.clear()

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))

        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)

        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        if len(self.deque) == 0:
            return "_NULL_"

        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
