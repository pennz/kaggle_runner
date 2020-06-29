import re
import random
import gc
import pandas as pd
import numpy as np
import gc
import cv2
import albumentations as A
from albumentations import (Blur, Compose, ElasticTransform, GaussNoise,
                            GridDistortion, HorizontalFlip, IAAEmboss,
                            Normalize, OneOf, #MultiplicativeNoise,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomGamma, RandomRotate90, Resize,
                            ShiftScaleRotate, Transpose, VerticalFlip)
from albumentations.pytorch import ToTensor
from albumentations.core.transforms_interface import DualTransform, BasicTransform
from kaggle_runner.utils.kernel_utils import get_obj_or_dump
from nltk import sent_tokenize

LANGS = {
    'en': 'english',
    'it': 'italian',
    'fr': 'french',
    'es': 'spanish',
    'tr': 'turkish',
    'ru': 'russian',
    'pt': 'portuguese'
}

class NLPTransform(BasicTransform):
    """ Transform for nlp task."""

    @property
    def targets(self):
        return {"data": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation

        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value

        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, LANGS.get(lang, 'english'))


class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """

    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)

        return ' '.join(sentences), lang


def get_open_subtitles():
    df_ot = get_pickled_data("ot.pkl")

    if df_ot is None:
        df_ot = pd.read_csv(f'{ROOT_PATH}/input/open-subtitles-toxic-pseudo-labeling/open-subtitles-synthesic.csv',
                            index_col='id')[['comment_text', 'toxic', 'lang']]
        df_ot = df_ot[~df_ot['comment_text'].isna()]
        df_ot['comment_text'] = df_ot.parallel_apply(
            lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
        df_ot = df_ot.drop_duplicates(subset='comment_text')
        df_ot['toxic'] = df_ot['toxic'].round().astype(np.int)
        get_obj_or_dump("ot.pkl", default=df_ot)

    return df_ot

ROOT_PATH = '/kaggle'
def get_pickled_data(file_path):
    obj = get_obj_or_dump(file_path)

    if obj is None:
        #may_debug(True)

        return get_obj_or_dump(f"{ROOT_PATH}/input/clean-pickle-for-jigsaw-toxicity/{file_path}")

    return obj

class SynthesicOpenSubtitlesTransform(NLPTransform):
    def __init__(self, always_apply=False, supliment_toxic=None, p=0.5, mix=False):
        super(SynthesicOpenSubtitlesTransform, self).__init__(always_apply, p)

        df = get_open_subtitles()
        self.synthesic_toxic = df[df['toxic'] == 1].comment_text.values
        self.synthesic_non_toxic = df[df['toxic'] == 0].comment_text.values

        if supliment_toxic is not None:
            self.synthesic_toxic = np.concatenate(
                (self.synthesic_toxic, supliment_toxic))
        self.mix = mix

        del df
        gc.collect()

    def _mix_both(self, texts):
        for i in range(random.randint(0, 2)):
            texts.append(random.choice(self.synthesic_non_toxic))

        for i in range(random.randint(1, 3)):
            texts.append(random.choice(self.synthesic_toxic))

    def generate_synthesic_sample(self, text, toxic):
        texts = [text]

        if toxic == 0:
            if self.mix:
                self._mix_both(texts)
                toxic = 1
            else:
                for i in range(random.randint(1, 5)):
                    texts.append(random.choice(self.synthesic_non_toxic))
        else:
            self._mix_both(texts)
        random.shuffle(texts)

        return ' '.join(texts), toxic

    def apply(self, data, **params):
        text, toxic = data
        text, toxic = self.generate_synthesic_sample(text, toxic)

        return text, toxic


def get_transforms(phase, size, mean, std):
    list_transforms = []

    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                GaussNoise(),
                #A.MultiplicativeNoise(multiplier=1.5, p=1),
            ]
        )
    list_transforms.extend(
        [Resize(size, size), Normalize(mean=mean, std=std, p=1), ToTensor(), ]
    )

    list_trfms = Compose(list_transforms)

    return list_trfms

# # + colab={} colab_type="code" id="8lcd0sZJ3bGj"


def get_sentences(text, lang='en'):
    return sent_tokenize(text, LANGS.get(lang, 'english'))

# # + colab={} colab_type="code" id="gMHg6Xvz3bGn"


def exclude_duplicate_sentences(text, lang='en'):
    sentences = []

    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()

        if sentence not in sentences:
            sentences.append(sentence)

    return ' '.join(sentences)

# # + colab={} colab_type="code" id="EizVrxLZ3bGr"


def clean_text(text, lang='en'):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = exclude_duplicate_sentences(text, lang)

    return text.strip()


# # + colab={} colab_type="code" id="ffbjaiBH3bGu"
# # + colab={} colab_type="code" id="3b2yIDOv3bG1"


class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """

    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform,
              self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = []

        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()

            if sentence not in sentences:
                sentences.append(sentence)

        return ' '.join(sentences), lang

# # + colab={} colab_type="code" id="kyDSPoUF3bG4"


class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """

    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

# # + colab={} colab_type="code" id="cAbXjyV63bG8"


class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """

    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

# # + colab={} colab_type="code" id="UQHHd_wo3bG_"


class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """

    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

# # + colab={} colab_type="code" id="8sv-ecgw3bHC"


class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """

    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang
