import os
from kaggle_runner import logger
from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.metrics.metrics import matthews_correlation
from kaggle_runner.datasets.transfomers import *
from kaggle_runner.datasets.bert import DatasetRetriever
from kaggle_runner.utils.kernel_utils import get_obj_or_dump
from kaggle_runner.modules.ToxicSimpleNNModel import ToxicSimpleNNModel
from fastai.basic_data import DataBunch
import transformers
from transformers import *
import albumentations

ROOT_PATH = f'/kaggle' # for colab
BACKBONE_PATH = 'xlm-roberta-large'

from kaggle_runner.defaults import SEED
import random
import numpy as np

def seed_everything(seed):
    """seed_everything.

    :param seed: number to generate pseudo random states
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_train_transforms():
    """get_train_transforms."""

    return albumentations.Compose([
        ExcludeUsersMentionedTransform(p=0.95),
        ExcludeUrlsTransform(p=0.95),
        ExcludeNumbersTransform(p=0.95),
        ExcludeHashtagsTransform(p=0.95),
        ExcludeDuplicateSentencesTransform(p=0.95),
    ], p=1.0)

def get_synthesic_transforms(supliment_toxic, p=0.5, mix=False):
    """get_synthesic_transforms.

    :param supliment_toxic: extra toxic data which will be used in synthesizing data
    :param p: probability
    :param mix: flag for mixing more data with toxic (it does not work with balanced sampler, too many toxic data)
    """

    return SynthesicOpenSubtitlesTransform(p=p, supliment_toxic=supliment_toxic, mix=mix)

ROOT_PATH = '/kaggle'
def get_pickled_data(file_path):
    """get_pickled_data from current folder or kaggle data input folder.

    :param file_path:
    """
    obj = get_obj_or_dump(file_path)

    if obj is None:
        #may_debug(True)

        return get_obj_or_dump(f"{ROOT_PATH}/input/clean-pickle-for-jigsaw-toxicity/{file_path}")

    return obj

vocab = get_pickled_data("vocab.pkl")
#if vocab is None: # vocab file read~~
#   vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]
#   get_obj_or_dump("vocab.pkl", default=vocab)

class Shonenkov(FastAIKernel):
    """Shonenkov kernel, use TPU to train; for data, down sampler and synthesizing."""

    def __init__(self, device, config, **kargs):
        """__init__.

        :param device:
        :param config:
        :param kargs:
        """
        super(Shonenkov, self).__init__(**kargs)
        self.data = None
        self.transformers = None
        self.setup_transformers()
        self.device = device
        self.config = config
        self.learner = None

    def build_and_set_model(self):
        """build_and_set_model."""
        self.model = ToxicSimpleNNModel()
        self.model = self.model.to(self.device)

    def set_random_seed(self):
        """set_random_seed."""
        seed_everything(SEED)

    def setup_transformers(self):
        """setup_transformers."""

        if self.transformers is None:
            supliment_toxic = None # avoid overfit
            train_transforms = get_train_transforms();
            synthesic_transforms_often = get_synthesic_transforms(supliment_toxic, p=0.5)
            synthesic_transforms_low = None
            tokenizer = XLMRobertaTokenizer.from_pretrained(BACKBONE_PATH)
            shuffle_transforms = ShuffleSentencesTransform(always_apply=True)

            self.transformers = {'train_transforms': train_transforms,
                                 'synthesic_transforms_often': synthesic_transforms_often,
                                 'synthesic_transforms_low': synthesic_transforms_low,
                                 'tokenizer': tokenizer, 'shuffle_transforms':
                                 shuffle_transforms}

    def prepare_train_dev_data(self):
        """prepare_train_dev_data."""
        df_train = get_pickled_data("train.pkl")

        if df_train is None:
            df_train = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-toxicity-train-data-with-aux/train_data.csv')
            df_train['comment_text'] = df_train.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
            get_obj_or_dump("train.pkl", default=df_train)

        #supliment_toxic = get_toxic_comments(df_train)
        self.train_dataset = DatasetRetriever(
            labels_or_ids=df_train['toxic'].values,
            comment_texts=df_train['comment_text'].values,
            langs=df_train['lang'].values,
            severe_toxic=df_train['severe_toxic'].values,
            obscene=df_train['obscene'].values,
            threat=df_train['threat'].values,
            insult=df_train['insult'].values,
            identity_hate=df_train['identity_hate'].values,
            use_train_transforms=True,
            transformers=self.transformers
        )
        df_val = get_pickled_data("val.pkl")

        if df_val is None:
            df_val = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', index_col='id')
            df_val['comment_text'] = df_val.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
            get_obj_or_dump("val.pkl", default=df_val)

        self.validation_tune_dataset = DatasetRetriever(
            labels_or_ids=df_val['toxic'].values,
            comment_texts=df_val['comment_text'].values,
            langs=df_val['lang'].values,
            use_train_transforms=True,
            transformers=self.transformers
        )
        self.validation_dataset = DatasetRetriever(
            labels_or_ids=df_val['toxic'].values,
            comment_texts=df_val['comment_text'].values,
            langs=df_val['lang'].values,
            use_train_transforms=False,
            transformers=self.transformers
        )

        del df_val
        gc.collect();

        del df_train
        gc.collect();

    def prepare_test_data(self):
        """prepare_test_data."""

        if os.path.exists('/content'): # colab
            df_test = get_pickled_data("test.pkl")
        else:
            df_test = None

        if df_test is None:
            df_test = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/test.csv', index_col='id')
            df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)
            get_obj_or_dump("test.pkl", default=df_test)

        self.test_dataset = DatasetRetriever(
            labels_or_ids=df_test.index.values, ## here different!!!
            comment_texts=df_test['comment_text'].values,
            langs=df_test['lang'].values,
            use_train_transforms=False,
            test=True,
            transformers=self.transformers
        )

        del df_test
        gc.collect();
    def after_prepare_data_hook(self):
        """Put to databunch here"""
        logger.debug("kernel use device %s", self.device)
        self.data = DataBunch.create(self.train_dataset,
                                     self.validation_dataset,
                                     bs=self.config.batch_size,
                                     device=self.device,
                                     num_workers=self.config.num_workers)

    def peek_data(self):
        """peek_data."""

        if self.data is not None:
            may_debug()
            o = self.data.one_batch()
            print(o)

            return o
        else:
            if self.logger is not None:
                self.logger.error("peek_data failed, DataBunch is None.")

from kaggle_runner import may_debug

import torch.nn as nn

class ToxicSimpleNNModelChangeInner(nn.Module):
    """ToxicSimpleNNModelChangeInner."""


    def __init__(self, use_aux=True):
        """__init__.

        :param use_aux:
        """
        super(ToxicSimpleNNModelChangeInner, self).__init__()

        self.backbone = XLMModel.from_pretrained('xlm-mlm-tlm-xnli15-1024')
        self.dropout = nn.Dropout(0.3)
        aux_len = 0

        if use_aux:
            aux_len = 5

        #in_features = self.backbone.layer[11].ff.layer_2.out_features*2
        #in_features = self.backbone.pooler.dense.out_features*2 # bert - xmlrobert
        in_features = self.backbone.ffns[11].lin2.out_features*2 # xlm
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=2+aux_len,
        )

    def forward(self, input_ids, attention_masks):
        """forward.

        :param input_ids:
        :param attention_masks:
        """
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(
            input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)

        return self.linear(x)

class ShonenkovChangeInner(Shonenkov):
    """ShonenkovChangeInner."""

    def __init__(self, device, config, **kargs):
        """__init__.

        :param device:
        :param config:
        :param kargs:
        """
        super(ShonenkovChangeInner, self).__init__(device, config, **kargs)
        assert self.transformers is not None

    def build_and_set_model(self):
        """build_and_set_model."""
        self.model = ToxicSimpleNNModelChangeInner()
        self.model = self.model.to(self.device)

    def prepare_train_dev_data(self):
        """prepare_train_dev_data."""
        df_train = get_pickled_data("train_XLM.pkl")

        if df_train is None:
            df_train = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-toxicity-train-data-with-aux/train_data.csv')
            df_train['comment_text'] = df_train.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
            get_obj_or_dump("train_XLM.pkl", default=df_train)

        #supliment_toxic = get_toxic_comments(df_train)
        self.train_dataset = DatasetRetriever(
            labels_or_ids=df_train['toxic'].values,
            comment_texts=df_train['comment_text'].values,
            langs=df_train['lang'].values,
            severe_toxic=df_train['severe_toxic'].values,
            obscene=df_train['obscene'].values,
            threat=df_train['threat'].values,
            insult=df_train['insult'].values,
            identity_hate=df_train['identity_hate'].values,
            use_train_transforms=True,
            transformers=self.transformers
        )
        df_val = get_pickled_data("val_XLM.pkl")

        if df_val is None:
            df_val = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', index_col='id')
            df_val['comment_text'] = df_val.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
            get_obj_or_dump("val_XLM.pkl", default=df_val)

        self.validation_tune_dataset = DatasetRetriever(
            labels_or_ids=df_val['toxic'].values,
            comment_texts=df_val['comment_text'].values,
            langs=df_val['lang'].values,
            use_train_transforms=True,
            transformers=self.transformers
        )
        self.validation_dataset = DatasetRetriever(
            labels_or_ids=df_val['toxic'].values,
            comment_texts=df_val['comment_text'].values,
            langs=df_val['lang'].values,
            use_train_transforms=False,
            transformers=self.transformers
        )

        del df_val
        gc.collect();

        del df_train
        gc.collect();

    def prepare_test_data(self):
        """prepare_test_data."""

        if os.path.exists('/content'): # colab
            df_test = get_pickled_data("test_XLM.pkl")
        else:
            df_test = None

        if df_test is None:
            df_test = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/test.csv', index_col='id')
            df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)
            get_obj_or_dump("test_XLM.pkl", default=df_test)

        self.test_dataset = DatasetRetriever(
            labels_or_ids=df_test.index.values, ## here different!!!
            comment_texts=df_test['comment_text'].values,
            langs=df_test['lang'].values,
            use_train_transforms=False,
            test=True,
            transformers=self.transformers
        )

        del df_test
        gc.collect();

    def setup_transformers(self):
        """setup_transformers."""

        if not hasattr(self, 'transformers') or self.transformers is None:
            supliment_toxic = None # avoid overfit
            train_transforms = get_train_transforms();
            synthesic_transforms_often = get_synthesic_transforms(supliment_toxic, p=0.5)
            synthesic_transforms_low = None
            shuffle_transforms = ShuffleSentencesTransform(always_apply=True)

            #from tokenizers import BertWordPieceTokenizer
            #tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024')

            self.transformers = {'train_transforms': train_transforms,
                                 'synthesic_transforms_often': synthesic_transforms_often,
                                 'synthesic_transforms_low': synthesic_transforms_low,
                                 'tokenizer': tokenizer,
                                 'shuffle_transforms': shuffle_transforms}

import torch
class DummyTrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs , 8 for GPU, 16 for TPU
    n_epochs = 2  # количество эпох для обучения
    lr = 0.3 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
    fold_number = 0  # номер фолда для обучения

    # -------------------
    verbose = True  # выводить принты
    verbose_step = 25  # количество шагов для вывода принта
    # -------------------

    # --------------------
    step_scheduler = False  # выполнять scheduler.step после вызова optimizer.step
    validation_scheduler = True  # выполнять scheduler.step после валидации loss (например для плато)
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # -------------------
    criterion = 'L'
    # -------------------


def test_change_XLM_module():
    """test_change_XLM_module."""
    from kaggle_runner.losses import LabelSmoothing
    k = ShonenkovChangeInner(torch.device("cpu"), DummyTrainGlobalConfig,
                             metrics=None, loss_func=LabelSmoothing(),
                             opt_func=None)
    assert k is not None
    k.run(dump_flag=False)
