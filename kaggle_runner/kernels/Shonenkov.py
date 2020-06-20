from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.metrics.metrics import matthews_correlation
from kaggle_runner.datasets.transfomers import *
from kaggle_runner.utils.kernel_utils import get_obj_or_dump
import albumentations

ROOT_PATH = f'/kaggle' # for colab

def get_train_transforms():
    return albumentations.Compose([
        ExcludeUsersMentionedTransform(p=0.95),
        ExcludeUrlsTransform(p=0.95),
        ExcludeNumbersTransform(p=0.95),
        ExcludeHashtagsTransform(p=0.95),
        ExcludeDuplicateSentencesTransform(p=0.95),
    ], p=1.0)

def get_synthesic_transforms(supliment_toxic, p=0.5, mix=False):
    return SynthesicOpenSubtitlesTransform(p=p, supliment_toxic=supliment_toxic, mix=mix)

def get_pickled_data(file_path):
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
    def __init__(self, device, **kargs):
        super(Shonenkov, self).__init__(**kargs)
        self.data = None
        self.transformers = None
        self.setup_transformers()
        self.device = device
        self.learner = None

    def build_and_set_model(self):
        self.model = ToxicSimpleNNModel()
        self.model = self.model.to(self.device)

    def set_random_seed(self):
        seed_everything(SEED)

    def setup_transformers(self):
        if self.transformers is None:
            supliment_toxic = None # avoid overfit
            train_transforms = get_train_transforms();
            synthesic_transforms_often = get_synthesic_transforms(supliment_toxic, p=0.5)
            synthesic_transforms_low = None
            #tokenizer = tokenizer
            shuffle_transforms = ShuffleSentencesTransform(always_apply=True)

            self.transformers = {'train_transforms': train_transforms,
                                 'synthesic_transforms_often': synthesic_transforms_often,
                                 'synthesic_transforms_low': synthesic_transforms_low,
                                 'tokenizer': tokenizer, 'shuffle_transforms':
                                 shuffle_transforms}

    def prepare_train_dev_data(self):
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
        df_test = get_pickled_data("test.pkl")

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
                                     bs=TrainGlobalConfig.batch_size,
                                     device=self.device,
                                     num_workers=TrainGlobalConfig.num_workers)

    def peek_data(self):
        if self.data is not None:
            may_debug()
            o = self.data.one_batch()
            print(o)

            return o
        else:
            if self.logger is not None:
                self.logger.error("peek_data failed, DataBunch is None.")
