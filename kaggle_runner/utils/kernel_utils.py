import errno
import logging
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import torch.distributed as dist
import torch.utils.data
from IPython.core.debugger import set_trace
from PIL import ImageFile
from tensorflow.python.ops import math_ops

from .utils import logger

trace_flag = True


ImageFile.LOAD_TRUNCATED_IMAGES = True


class AMQPURL:
    class AMQPURL_DEV:
        host = "termite.rmq.cloudamqp.com"  # (Load balanced)
        passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"
        username = "drdsfaew"
        Vhost = "drdsfaew"
        # host = "127.0.0.1"  # (Load balanced)
        # passwd = "guest"
        # username = "guest"
        # Vhost = "/"

    def __init__(
        self,
        host=AMQPURL_DEV.host,
        passwd=AMQPURL_DEV.passwd,
        Vhost=AMQPURL_DEV.Vhost,
        username=AMQPURL_DEV.username,
    ):
        self.host = host
        self.passwd = passwd
        self.Vhost = Vhost
        self.username = username

    def string(self):
        Vhost = self.Vhost

        if self.Vhost == "/":
            Vhost = ""

        return f"amqp://{self.username}:{self.passwd}@{self.host}/{Vhost}"


BIN_FOLDER = (
    "/content/gdrivedata/My Drive/" if os.path.isdir(
        "/content/gdrivedata") else "./"
)

DATASET_BASE_FOLDER = '/kaggle/input'

def dump_obj(obj, filename, fullpath=False, force=False):
    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename

    if not force and os.path.isfile(path):
        logger.debug(f"{path} already existed, not dumping")
    else:
        logger.debug(f"Overwrite {path}!")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

def get_kaggle_dataset_input(filename):
    filename = os.path.join(DATASET_BASE_FOLDER, filename)

    return get_obj_or_dump(filename, fullpath=True, default=None)

def get_obj_or_dump(filename, fullpath=False, default=None):
    """get_obj_or_dump will dump default obj to file if file not there, otherwise
    obj will be unpickled from file. If file not found, default value will returned."""

    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename

    if os.path.isfile(path):
        logger.debug("load " + filename)
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        if default is not None:
            logger.debug("dump :" + filename)
            dump_obj(default, filename)

        return default


def file_exist(filename, fullpath=False):
    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename

    return os.path.isfile(path)


# 0.5 means no rebalance
def binary_crossentropy_with_focal_seasoned(
    y_true, logit_pred, beta=0.0, gamma=1.0, alpha=0.5, custom_weights_in_Y_true=True
):
    """
    :param alpha:weight for positive classes **loss**. default to 1- true
        positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or
        hyperparameter.
    :param custom_weights_in_Y_true:
    :return:
    """
    balanced = gamma * logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    # only use gamma in this layer, easier to split out factor

    return binary_crossentropy_with_focal(
        y_true,
        y_pred,
        gamma=0,
        alpha=alpha,
        custom_weights_in_Y_true=custom_weights_in_Y_true,
    )


# 0.5 means no rebalance
def binary_crossentropy_with_focal(
    y_true, y_pred, gamma=1.0, alpha=0.5, custom_weights_in_Y_true=True
):
    """
    https://arxiv.org/pdf/1708.02002.pdf

    $$ FL(p_t) = -(1-p_t)^{\gamma}log(p_t) $$
    $$ p_t=p\: if\: y=1$$
    $$ p_t=1-p\: otherwise$$

    :param y_true:
    :param y_pred:
    :param gamma: make easier ones weights down
    :param alpha: weight for positive classes. default to 1 - (true
        positive cnts / all cnts), alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practice α may be set by inverse class freqency or
        hyperparameter.
    :return: bce
    """
    # assert 0 <= alpha <= 1 and gamma >= 0
    # hyper parameters, just use the one for binary?
    # alpha = 1. # maybe smaller one can help, as multi-class will make the
    # error larger
    # gamma = 1.5 # for our problem, try different gamma

    # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
    #       bce = target * alpha* (1-output+epsilon())**gamma * math_ops.log(output + epsilon())
    #       bce += (1 - target) *(1-alpha)* (output+epsilon())**gamma * math_ops.log(1 - output + epsilon())
    # return -bce # binary cross entropy
    eps = tf.keras.backend.epsilon()

    if custom_weights_in_Y_true:
        custom_weights = y_true[:, 1:2]
        y_true = y_true[:, :1]

    if 1.0 - eps <= gamma <= 1.0 + eps:
        bce = alpha * math_ops.multiply(
            1.0 - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        )
        bce += (1 - alpha) * math_ops.multiply(
            y_pred, math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps))
        )
    elif 0.0 - eps <= gamma <= 0.0 + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply(
            (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)
        )
    else:
        gamma_tensor = tf.broadcast_to(
            tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(
            math_ops.pow(1.0 - y_pred, gamma_tensor),
            math_ops.multiply(y_true, math_ops.log(y_pred + eps)),
        )
        bce += (1 - alpha) * math_ops.multiply(
            math_ops.pow(y_pred, gamma_tensor),
            math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)),
        )

    if custom_weights_in_Y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce


def reinitLayers(model):
    session = K.get_session()

    for layer in model.layers:
        # if isinstance(layer, keras.engine.topology.Container):

        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)

            continue
        print("LAYER::", layer.name)

        if layer.trainable is False:
            continue

        for v in layer.__dict__:
            v_arg = getattr(layer, v)

            if hasattr(v_arg, "initializer"):
                # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, "initializer")
                initializer_method.run(session=session)
                print("reinitializing layer {}.{}".format(layer.name, v))


# Evaluation metric
# ref https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data
def dice_coef(y_true, y_pred, smooth=1, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_pred_b = math_ops.cast(y_pred_f > threshold, y_pred.dtype)
    y_true_b = math_ops.cast(y_true_f > threshold, y_pred.dtype)

    intersection = K.sum(y_true_b * y_pred_b)

    return (2.0 * intersection + smooth) / (K.sum(y_true_b) + K.sum(y_pred_b) + smooth)


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]

            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0

    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []

    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all
    processes have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()

    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters

        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False

        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]

            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " " + " ".join(rle)


def online_mean_and_sd(loader, data_map=None):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    credit xwkuang5
    @https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/7
    """
    cnt = 0
    # fst_moment = torch.empty(3)
    # snd_moment = torch.empty(3)
    fst_moment = np.zeros(3)
    snd_moment = np.zeros(3)

    for data in loader:
        if data_map is not None:
            data = data_map(data)
        data = np.array([t.numpy() for t in data])
        b, c, h, w = data.shape  # data here is tuple... if loader batch > 1
        nb_pixels = b * h * w
        # sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_ = data.sum(axis=0).sum(axis=-1).sum(axis=-1)
        # sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        sum_of_square = (data ** 2).sum(axis=0).sum(axis=-1).sum(axis=-1)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, np.sqrt(snd_moment - fst_moment ** 2)


def download_file_one_at_a_time(file_name, directory=".", overwrite=True):
    if overwrite:
        run_process_print(
            'wget http://23.105.212.181:8000/{0} -O "{1}/{0}"'.format(
                file_name, directory
            )
        )
    else:
        run_process_print(
            "[ -f {1}/{0} ] || wget http://23.105.212.181:8000/{0} -P {1}".format(
                file_name, directory
            )
        )


def run_process_print(command_str):
    print(command_str.split())
    get_ipython().system(command_str)
