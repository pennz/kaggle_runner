import argparse
import copy
import gc
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint)
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization,
                                     Bidirectional, Dense, Embedding,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Lambda, Layer,
                                     PReLU, SpatialDropout1D, add, concatenate)
from tensorflow.keras.metrics import binary_crossentropy, mean_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops

import kaggle_runner.datasets.data_prepare as d
import kaggle_runner.utils.kernel_utils as utils
from kaggle_runner.defaults import *
from kaggle_runner.logs import NBatchProgBarLogger
# from gradient_reversal_keras_tf.flipGradientTF import GradientReversal
from kaggle_runner.losses import (binary_crossentropy_with_focal,
                                  binary_crossentropy_with_focal_seasoned)
from kaggle_runner.metrics.metrics import (binary_sensitivity,
                                           binary_sensitivity_np,
                                           binary_specificity)
from kaggle_runner.modules.attention import AttentionRaffel
from kaggle_runner.utils.kernel_utils import reinitLayers

NUM_MODELS = 2  # might be helpful but...

# BATCH_SIZE = 2048 * 2  # for cloud server runing
BATCH_SIZE = 1024 // 2  # for cloud server runing
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
RES_DENSE_HIDDEN_UNITS = 5

EPOCHS = (
    7  # 4 seems good for current setting, more training will help for the final score?
)

DEBUG_TRACE = False
# all for debug
preds = None
kernel = None
X, y, idx_train_background, idx_val_background = None, None, None, None
y_res_pred = None

# lambda e: 5e-3 * (0.5 ** e),
STARTER_LEARNING_RATE = 1e-3  # as the BCE we adopted...
LEARNING_RATE_DECAY_PER_EPOCH = 0.5

IDENTITY_RUN = True
TARGET_RUN = "lstm"
TARGET_RUN_READ_RESULT = False
RESTART_TRAIN = False
RESTART_TRAIN_RES = True
RESTART_TRAIN_ID = False

NO_AUX = True
Y_TRAIN_BIN = False  # with True, slightly worse

FOCAL_LOSS = False

PRD_ONLY = True  # will not train the model
NOT_PRD = True

FOCAL_LOSS_GAMMA = 0.0
FINAL_SUBMIT = True
FINAL_DEBUG = True

if FINAL_SUBMIT:
    TARGET_RUN = "lstm"
    EPOCHS = 6
    PRD_ONLY = False  # not training
    NOT_PRD = False  # prd submission test
    RESTART_TRAIN = True
    FOCAL_LOSS_GAMMA = 0.0

    DEBUG_TRACE = False

    if FINAL_DEBUG:
        EPOCHS = 4
        DEBUG_TRACE = False
        PRD_ONLY = False  # not training if True
        RESTART_TRAIN = False
        NOT_PRD = True  # prd submission test
        NUM_MODELS = 1  # might timeout, so back to 2
        STARTER_LEARNING_RATE = 1e-3
        Y_TRAIN_BIN = False


# Credits for https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043


# TODO fix the large class
class KaggleKernel:
    def __init__(self, action=None):
        self.model = None
        self.emb = None
        self.train_X = None
        self.train_X_all = None
        self.train_y_all = None
        self.train_y = None
        self.train_y_aux = None
        self.train_y_aux_all = None
        self.train_y_identity = None
        self.train_X_identity = None
        # self.to_predict_X = None
        self.embedding_matrix = None
        self.embedding_layer = None
        self.identity_idx = None
        self.id_used_in_train = False
        self.id_validate_df = None

        self.judge = None  # for auc metrics

        self.oof_preds = None
        self.load_data(action)

    def load_data(self, action):
        if self.emb is None:
            self.emb = d.EmbeddingHandler()
        self.emb.read_train_test_df(train_only=False)
        self.train_df = self.emb.train_df

        if PRD_ONLY or TARGET_RUN_READ_RESULT or ANA_RESULT:
            pass
        else:  # keep record training parameters
            utils.dump_obj(
                f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}",
                "run_info.txt",
                force=True,
            )

        (
            self.train_X_all,
            self.train_y_all,
            self.train_y_aux_all,
            self.to_predict_X_all,
            self.embedding_matrix,
        ) = self.emb.data_prepare(action)
        # picked ones are binarized
        self.train_y_aux_all = self.train_df[d.AUX_COLUMNS].values

        if Y_TRAIN_BIN:
            self.train_y_float_backup = self.train_y_all
            self.train_y_all = np.where(self.train_y_all >= 0.5, True, False)
        # if action is not None:
        #    if action == TRAIN_DATA_EXCLUDE_IDENDITY_ONES:
        self.load_identity_data_idx()
        mask = np.ones(len(self.train_X_all), np.bool)
        # identities data excluded first ( will add some back later)
        mask[self.identity_idx] = 0

        # need get more 80, 000 normal ones without identities, 40,0000 %40 with identities, 4*0.4/12, 0.4/3 13%
        add_no_identity_to_val = self.train_df[mask].sample(
            n=int(8e4), random_state=2019
        )
        add_no_identity_to_val_idx = add_no_identity_to_val.index
        mask[add_no_identity_to_val_idx] = 0  # exclude from train, add to val

        self.train_mask = mask

        self.train_X = self.train_X_all[mask]
        self.train_y = self.train_y_all[mask]
        self.train_y_aux = self.train_y_aux_all[mask]
        logger.debug("Train data no identity ones now")

        try:
            if self.emb.do_emb_matrix_preparation:
                exit(0)  # saved and can exit
        except:
            logger.warning(
                "Prepare emb for embedding error, so we might already have process file and load data, and we continue"
            )

    # @staticmethod
    # def bin_prd_clsf_info(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-12):
    #    """
    #    refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

    #    Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
    #     of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

    #    1) If N >> P, f1 is a better.

    #    2) If P >> N, b_acc is better.

    #    For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

    #    :param y_true:
    #    :param y_pred:
    #    :param threshold:
    #    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    #    """
    #    if FOCAL_LOSS_GAMMA == 2.0:
    #        threshold = 0.57
    #    elif FOCAL_LOSS_GAMMA == 1.0:
    #        threshold = (0.53+(0.722-0.097))/2  #(by...reading the test result..., found it changes every training... so useless)
    #    threshold = math_ops.cast(threshold, y_pred.dtype)
    #    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    #    y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

    #    ground_pos = math_ops.reduce_sum(y_true) + epsilon
    #    correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
    #    predict_pos = math_ops.reduce_sum(y_pred) + epsilon

    #    precision = math_ops.div(correct_pos, predict_pos)
    #    recall = math_ops.div(correct_pos, ground_pos)

    #    if N_MORE:
    #        m = (2*recall*precision) / (precision+recall)
    #    else:
    #        #m = (sensitivity + specificity)/2 # balanced accuracy
    #        raise NotImplementedError("Balanced accuracy metric is not implemented")

    #    return m

    # so nine model for them...

    def build_lstm_model_customed(
        self,
        num_aux_targets,
        embedding_matrix,
        embedding_layer=None,
        with_aux=False,
        loss="binary_crossentropy",
        metrics=None,
        hidden_act="relu",
        with_BN=False,
    ):
        """build lstm model, non-binarized

        cls_reg: 0 for classification, 1 for regression(linear)
        metrics: should be a list
        with_aux: default False, don't know how to interpret...
        with_BN: ... BatchNormalization not work, because different subgroup different info, batch normalize will make it worse?
        """
        logger.debug(
            f"model detail: loss {loss}, hidden_act {hidden_act}, with_BN {with_BN}"
        )

        if num_aux_targets > 0 and not with_aux:
            raise RuntimeError(
                "aux features numbers given but aux not enabled")

        if num_aux_targets <= 0 and with_aux:
            raise RuntimeError("aux features numbers invalid when aux enabled")

        words = Input(shape=(d.MAX_LEN,))  # (None, 180)

        if embedding_layer is not None:
            x = embedding_layer(words)
        else:
            x = Embedding(
                *embedding_matrix.shape, weights=[embedding_matrix], trainable=False
            )(words)

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate(
            [
                AttentionRaffel(d.MAX_LEN, name="attention_after_lstm")(x),
                GlobalMaxPooling1D()(x),  # with this 0.9125 ...(not enough test...)
                # GlobalAveragePooling1D()(x),  # a little worse to use this, 0.9124
            ]
        )

        activate_type = hidden_act

        if activate_type == "prelu":  # found it not working
            hidden = add(
                [hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))]
            )

            if with_BN:
                hidden = BatchNormalization()(hidden)
            hidden = add(
                [hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))]
            )
        else:
            hidden = add(
                [hidden, Dense(DENSE_HIDDEN_UNITS,
                               activation=activate_type)(hidden)]
            )

            if with_BN:
                hidden = BatchNormalization()(hidden)
            hidden = add(
                [hidden, Dense(DENSE_HIDDEN_UNITS,
                               activation=activate_type)(hidden)]
            )

        logit = Dense(1, activation=None)(hidden)
        logit = Lambda(
            lambda pre_sigmoid: FOCAL_LOSS_GAMMA_NEG_POS * pre_sigmoid
            + FOCAL_LOSS_BETA_NEG_POS
        )(logit)
        result = Activation("sigmoid")(logit)

        if with_aux:
            aux_result = Dense(num_aux_targets, activation="sigmoid")(hidden)
            model = Model(inputs=words, outputs=[result, aux_result])
            model.compile(
                loss=[loss, "binary_crossentropy"],
                optimizer="adam",
                loss_weights=[1.0, 0.25],
                metrics=metrics,
            )
        else:
            model = Model(inputs=words, outputs=result)
            model.compile(loss=loss, optimizer="adam", metrics=metrics)

        return model

    def build_lstm_model(self, num_aux_targets):
        words = Input(shape=(None,))
        x = Embedding(
            *self.embedding_matrix.shape,
            weights=[self.embedding_matrix],
            trainable=False,
        )(words)
        logger.info(
            "Embedding fine, here the type of embedding matrix is {}".format(
                type(self.embedding_matrix)
            )
        )

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate(
            [GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x), ])
        hidden = add(
            [hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])
        hidden = add(
            [hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])
        result = Dense(1, activation="sigmoid")(hidden)
        aux_result = Dense(num_aux_targets, activation="sigmoid")(hidden)

        model = Model(inputs=words, outputs=[result, aux_result])
        # model = Model(inputs=words, outputs=result_with_aux)
        # model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(0.005),
            metrics=[KaggleKernel.bin_prd_clsf_info],
        )
        # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
        #       bce = target * math_ops.log(output + epsilon())
        #       bce += (1 - target) * math_ops.log(1 - output + epsilon())
        # return -bce # binary cross entropy

        # for binary accuraty:
        # def binary_accuracy(y_true, y_pred, threshold=0.5):
        #   threshold = math_ops.cast(threshold, y_pred.dtype)
        #   y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
        #   return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

        # fit a line for linear regression, we use least square error, (residuals), norm, MLE
        # logistic regression uses the log(odds) on the y-axis, as log(odds) push points to
        # + , - infinity, we cannot use least square error, we use maximum likelihood, the line
        # can still be imagined to there. for every w guess, we have a log-likelihood for the
        # line. we need to find the ML line

        return model

    def run_lstm_model(
            self,
            final_train=False,
            n_splits=NUM_MODELS,
            predict_ones_with_identity=True,
            params=None,
            val_mask=None,
    ):
        # TODO condiser use train test split as
        """
        train_df, validate_df = model_selection.train_test_split(train,
        test_size=0.2)
        logger.info('%d train comments, %d validate comments' % (len(train_df),
        len(validate_df)))"""

        # self.oof_preds = np.zeros((self.train_X.shape[0], 1 + self.train_y_aux.shape[1]))
        test_preds = np.zeros((self.to_predict_X_all.shape[0]))
        prefix = params["prefix"]
        re_train = params["re-start-train"]
        predict_only = params["predict-only"]
        logger.debug(f"pred only? {predict_only}\n\n\n")
        # sample_weights = params.get('sample_weights')
        train_data = params.get("train_data", None)
        train_y_aux_passed = params.get("train_y_aux")
        val_y_aux_passed = params.get("val_y_aux")
        patience = params.get("patience", 3)
        bin_target = params.get("binary")
        f_g = params.get("fortify_subgroup")

        if train_data is None:
            train_X = self.train_X
            train_y = self.train_y
            train_y_aux = self.train_y_aux
        else:
            train_X, train_y = train_data
            train_y_aux = train_y_aux_passed
            val_y_aux = val_y_aux_passed

        if train_y_aux is None and not NO_AUX:
            raise RuntimeError("Need aux labels to train")

        if val_y_aux is None and not NO_AUX:
            raise RuntimeError("Need aux labels to validate")

        val_data = params.get("val_data")

        if val_data is None:  # used to dev evaluation
            val_X = self.train_X_identity
        else:
            val_X, val_y = val_data

        prefix += "G{:.1f}".format(FOCAL_LOSS_GAMMA)

        if self.embedding_layer is None:
            self.embedding_layer = Embedding(
                *self.embedding_matrix.shape,
                weights=[self.embedding_matrix],
                trainable=False,
            )

            del self.embedding_matrix
            gc.collect()

        # build one time, then reset if needed

        if NO_AUX:
            # we could load with file name, then remove and save to new one
            h5_file = prefix + "_attention_lstm_NOAUX_" + f".hdf5"
        else:
            # we could load with file name, then remove and save to new one
            h5_file = prefix + "_attention_lstm_" + f".hdf5"

        logger.debug(h5_file)

        starter_lr = params.get("starter_lr", STARTER_LEARNING_RATE)
        # model thing
        load_from_model = False

        if re_train or not os.path.isfile(h5_file):
            logger.debug(
                f"re_train is {re_train}, file {h5_file} exists? {os.path.isfile(h5_file)}"
            )

            if NO_AUX:
                if FOCAL_LOSS:
                    model = self.build_lstm_model_customed(
                        0,
                        None,
                        self.embedding_layer,
                        with_aux=False,
                        loss=binary_crossentropy_with_focal,
                        metrics=[binary_crossentropy, mean_absolute_error, ],
                    )
                else:
                    model = self.build_lstm_model_customed(
                        0,
                        None,
                        self.embedding_layer,
                        with_aux=False,
                        metrics=[mean_absolute_error, ],
                    )
            else:
                logger.debug(
                    "using loss=binary_crossentropy_with_focal_seasoned")
                model = self.build_lstm_model_customed(
                    len(self.train_y_aux[0]),
                    None,
                    self.embedding_layer,
                    with_aux=True,
                    loss=binary_crossentropy_with_focal,
                    metrics=[binary_crossentropy, mean_absolute_error],
                )
            self.model = model
            model.summary()
            logger.info("build model -> done\n\n\n")

        else:
            model = load_model(
                h5_file,
                custom_objects={
                    "binary_crossentropy_with_focal_seasoned": binary_crossentropy_with_focal_seasoned,
                    "binary_crossentropy_with_focal": binary_crossentropy_with_focal,
                    "AttentionRaffel": AttentionRaffel,
                },
            )
            load_from_model = True
            starter_lr = starter_lr * LEARNING_RATE_DECAY_PER_EPOCH ** (EPOCHS)
            self.model = model
            logger.debug(
                "restore from the model file {} -> done\n\n\n".format(h5_file))

        if FINAL_DEBUG and load_from_model:
            pred = model.predict(val_X, verbose=2, batch_size=BATCH_SIZE)
            self.check_preds_in_val(pred, val_mask, run_times=0)

        run_times = 0
        better_run_times = 0
        final_score = 0

        for fold in range(n_splits):
            # K.clear_session()  # so it will start over

            if fold > 0:
                reinitLayers(model)
                starter_lr = starter_lr / 8  # as lstm won't be re-initialized

            ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
            early_stop = EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=patience,
                restore_best_weights=True,
            )

            if not predict_only:
                prog_bar_logger = NBatchProgBarLogger(
                    display_per_batches=int(len(train_X) / 10 / BATCH_SIZE),
                    early_stop=True,
                    patience_displays=3,
                )

                if (
                    f_g is not None and bin_target
                ):  # only binary for subgroup, just try to see if it helps
                    # we also need to filter out the subgroup data (in outer layer now)
                    val_y = val_y >= 0.5

                sample_weights = None

                if train_y.shape[1] > 1:
                    _train_y = train_y[:,0]
                    sample_weights = train_y[:,1]
                    train_y = _train_y

                train_labels = train_y if NO_AUX else [train_y, train_y_aux]
                val_labels = val_y if NO_AUX else [val_y, val_y_aux]

                model.fit(
                    train_X,
                    train_labels,
                    # validation_split=val_split,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    sample_weight=sample_weights,
                    # steps_per_epoch=int(len(self.train_X)*False95/BATCH_SIZE),
                    verbose=0,  # prog_bar_logger will handle log, so set to 0 here
                    # validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                    validation_data=(val_X, val_labels),
                    callbacks=[
                        LearningRateScheduler(
                            # STARTER_LEARNING_RATE = 1e-2
                            # LEARNING_RATE_DECAY_PER_EPOCH = 0.7
                            lambda e: starter_lr * \
                                (LEARNING_RATE_DECAY_PER_EPOCH ** e),
                            verbose=1,
                        ),
                        early_stop,
                        ckpt,
                        prog_bar_logger,
                    ],
                )

            pred = model.predict(val_X, verbose=2, batch_size=BATCH_SIZE)
            final_score_this_run = self.check_preds_in_val(
                pred, val_mask, run_times=run_times
            )
            improve_flag = False

            if final_score_this_run > final_score:
                final_score = final_score_this_run
                improve_flag = True
            else:
                improve_flag = False

            if not NOT_PRD:
                # if final_train:
                test_result = model.predict(self.to_predict_X_all, verbose=2)
                run_times += 1

                if NO_AUX:
                    test_result_column = np.array(test_result).ravel()
                else:
                    # the shape of preds, is [0] is the predict,[1] for aux
                    test_result_column = np.array(test_result[0]).ravel()
                self.save_result(
                    test_result_column, filename=f"submission_split_{run_times}.csv"
                )

                if improve_flag:
                    logger.debug(
                        "better this time, will update submission.csv\n\n\n")
                    better_run_times += 1
                    test_preds += test_result_column
                    test_preds_avg = test_preds / better_run_times
                    # save everything in case failed in some folds
                    self.save_result(test_preds_avg)

    def check_preds_in_val(self, pred, val_mask, run_times=0):
        if DEBUG_TRACE:
            set_trace()

        if not NO_AUX:
            preds = np.array(pred[0]).ravel()
        else:
            preds = np.array(pred).ravel()

        if DEBUG_TRACE:
            set_trace()
        # preds = 1 / (1+np.exp(-preds))

        self.train_df.loc[val_mask, "lstm"] = preds

        if True:
            self.train_df[d.VAL_ERR_COLUMN] = 0
            # self.train_df.loc[val_mask, 'lstm'] = preds
            self.train_df.loc[val_mask, d.VAL_ERR_COLUMN] = (
                preds - self.train_df.loc[val_mask, d.TARGET_COLUMN]
            )
            dstr = self.target_analyzer.get_err_distribution(
                self.train_df, val_mask)

            for k, v in dstr.items():
                logger.debug(f"error info: {k}, {v[1]}")

        return self.calculate_metrics_and_print(
            filename_for_print=f"metrics_log_{run_times}.txt",
            validate_df_with_preds=kernel.train_df[val_mask],
            benchmark_base=kernel.train_df[val_mask],
        )

    #        test_preds /= run_times
    #        self.save_result(test_preds)

    def save_result(self, predictions, filename=None):
        if self.emb.test_df_id is None:
            self.emb.read_train_test_df()
        submission = pd.DataFrame.from_dict(
            {
                "id": self.emb.test_df_id,
                # 'id': test_df.id,
                "prediction": predictions,
            }
        )

        if filename is not None:
            submission.to_csv(filename, index=False)
        else:
            submission.to_csv("submission.csv", index=False)

    def prepare_second_stage_data_index(
        self, y_pred, subgroup="white", only_false_postitive=False, n_splits=5
    ):
        """

        :param y_pred: # might needed, compare pred and target, weight the examples
        :param subgroup:
        :param only_false_postitive: only train the ones with higher then actual ones?
        :return: X, y pair
        """
        df = self.id_validate_df
        subgroup_df = df[df[subgroup]]
        subgroup_idx = subgroup_df.index
        # size = len(subgroup_idx)

        splits = list(
            KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(
                subgroup_idx
            )
        )  # just use sklearn split to get id and it is fine. For text thing,
        # just do 1 fold, later we can add them all back
        tr_ind, val_ind = splits[0]

        return subgroup_df.iloc[tr_ind].index, subgroup_df.iloc[val_ind].index

    def build_res_model(
        self,
        subgroup,
        loss="binary_crossentropy",
        metrics=None,
        hidden_act="relu",
        with_BN=False,
    ):
        if self.model is None:
            logger.debug("Start loading model")
            h5_file = "/proc/driver/nvidia/G2.0_attention_lstm_NOAUX_0.hdf5"
            self.model = load_model(
                h5_file,
                custom_objects={
                    "binary_crossentropy_with_focal": binary_crossentropy_with_focal,
                    "AttentionRaffel": AttentionRaffel,
                },
            )
            logger.debug("Done loading model")
        base_model = self.model

        # add_1 = base_model.get_layer('add_1')
        add_2 = base_model.get_layer("add_2")
        main_logit = base_model.get_layer("dense_3")

        #       after add_1, add another dense layer, (so totally not change exited structure(might affect other subgroup?)
        #       then add this with add_2, finally go to sigmoid function (loss function no change ....) (should use small learning rate)
        # hidden = concatenate([
        #    add_2.output,
        #    Dense(RES_DENSE_HIDDEN_UNITS, activation=hidden_act, name="res_features_recombination")(add_1.output)
        # ], name='cat_'+subgroup+'_res_to_main')
        # result = Dense(1, activation='sigmoid', name='res_main_together')(hidden) -> not good
        res_recombination = Dense(
            RES_DENSE_HIDDEN_UNITS,
            activation=hidden_act,
            name="res_features_recombination",
        )(add_2.output)
        hidden = add(
            [
                res_recombination,
                Dense(RES_DENSE_HIDDEN_UNITS, activation=hidden_act, name="res_res")(
                    res_recombination
                ),
            ],
            name="res_res_added",
        )
        res_logit = Dense(1, activation=None, name="res_logit")(hidden)
        logit = add([res_logit, main_logit.output], name="whole_logit")
        result = Activation("sigmoid", name="whole_predict")(logit)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers

        for layer in base_model.layers:
            layer.trainable = False
        # compile the model (should be done *after* setting layers to non-trainable)
        model = Model(inputs=base_model.input, outputs=result)

        model.compile(optimizer="adam", loss=loss, metrics=metrics)

        return model

    # TODO put to model Trainer (subclass)
    def run_model_train(
        self, model, X, y, params, use_split=False, n_splits=5, y_aux=None
    ):
        if use_split:
            assert n_splits > 1
            # just use sklearn split to get id and it is fine. For text thing,
            splits = list(
                KFold(n_splits=n_splits, random_state=2019,
                      shuffle=True).split(X)
            )
        else:
            n_splits = 1  # for the following for loop

        prefix = params["prefix"]
        starter_lr = params["starter_lr"]
        validation_split = params.get("validation_split", 0.05)
        epochs = params.get("epochs", EPOCHS)
        lr_decay = params.get("lr_decay", LEARNING_RATE_DECAY_PER_EPOCH)
        patience = params.get("patience", 3)
        display_per_epoch = params.get("display_per_epoch", 5)
        display_verbose = params.get("verbose", 2)
        no_check_point = params.get("no_check_point", False)
        passed_check_point_file_path = params.get("check_point_path", None)

        prefix += "G{:.1f}".format(FOCAL_LOSS_GAMMA)

        for fold in range(n_splits):  # will need to do this later
            # K.clear_session()  # so it will start over todo fix K fold

            if use_split:
                tr_ind, val_ind = splits[fold]
                logger.info(
                    "%d train comments, %d validate comments" % (
                        tr_ind, val_ind)
                )

            else:
                tr_ind, val_ind = [True] * len(X), [False] * len(X)

            if NO_AUX:
                # we could load with file name, then remove and save to new one
                h5_file = prefix + "_attention_lstm_NOAUX_" + f"{fold}.hdf5"
            else:
                # we could load with file name, then remove and save to new one
                h5_file = prefix + "_attention_lstm_" + f"{fold}.hdf5"

            if passed_check_point_file_path is not None:
                h5_file = passed_check_point_file_path

            logger.debug(f"using checkpoint files: {h5_file}")

            early_stop = EarlyStopping(
                monitor="val_loss", mode="min", verbose=1, patience=patience
            )

            # data thing

            if NO_AUX:
                y_train = y[tr_ind]
                y_val = y[val_ind]
            else:
                y_train = [y[tr_ind], y_aux[tr_ind]]
                y_val = [y[val_ind], y_aux[val_ind]]

            callbacks = [
                LearningRateScheduler(
                    lambda e: starter_lr * (lr_decay ** e), verbose=1
                ),
                early_stop,
            ]

            if not no_check_point:
                ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
                callbacks.append(ckpt)

            if display_verbose == 1:
                verbose = 0
                prog_bar_logger = NBatchProgBarLogger(
                    display_per_batches=int(
                        len(tr_ind) / display_per_epoch / BATCH_SIZE
                    ),
                    early_stop=True,
                    patience_displays=patience,
                )
                callbacks.append(prog_bar_logger)
            else:  # 0 or 2
                verbose = display_verbose

            logger.debug(
                f"{len(tr_ind)} training, {validation_split*len(tr_ind)} validation in fit"
            )

            model.fit(
                X[tr_ind],
                y[tr_ind],
                validation_split=validation_split,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
            )

    def run_model(self, model, mode, X, y=None, params={}, n_splits=0):
        # checkpoint_predictions = []
        # weights = []

        if mode == "train":
            self.run_model_train(model, X, y, params, n_splits > 1, n_splits)
        elif mode == "predict":
            pred = model.predict(X, verbose=2, batch_size=BATCH_SIZE)

            return pred

        # if predict_ones_with_identity:
        #    return model.predict(self.train_X_identity, verbose=2, batch_size=BATCH_SIZE)

    def locate_data_in_np_train(self, index):
        """

        :param index: must be for the index of range 405130
        :return:
        """

        return self.train_X[index], self.train_y[index], self.train_y_aux[index]

    def locate_subgroup_index_in_np_train(self, subgroup):
        df = self.id_validate_df
        index = df[df[subgroup]].index

        return index

    def locate_subgroup_data_in_np_train(self, subgroup):
        """

        :param index: must be for the index of range 405130
        :return:
        """
        index = self.locate_subgroup_index_in_np_train(subgroup)

        return self.locate_data_in_np_train(index)

    def to_identity_index(self, index):
        """

        :param index: from 1.8m range index
        :return: to 0.4 m range index in identity data
        """
        df = self.id_validate_df

        # selected the items

        return [df.index.get_loc(label) for label in index]

    def _get_identities(self):
        """
        No need to use this function, all identities are marked

        :return:
        """
        prefix = self.emb.BIN_FOLDER
        # if os.path.isfile(prefix+'train_df.pd'):

        if False:
            self.train_df = pickle.load(open(prefix + "train_df.pd", "rb"))
        else:
            for g in d.IDENTITY_COLUMNS:
                pred = pickle.load(open(f"{prefix}_{g}_pred.pkl", "rb"))
                self.train_df[f"{g}_pred"] = pred

        for g in d.IDENTITY_COLUMNS:
            self.train_df.loc[self.identity_idx, f"{g}_pred"] = self.train_df.loc[
                self.identity_idx, g
            ]

    def get_identities_for_training(self):
        if not self.id_used_in_train:
            #            if not FINAL_SUBMIT:
            # in test set, around 10% data will be with identities (lower than training set)
            logger.debug("Use 90% identity data")
            id_df = self.train_df.loc[self.identity_idx]
            # 40,000 remained for val
            id_train_df = id_df.sample(frac=0.9, random_state=2019)
            id_train_df_idx = id_train_df.index
            #            else:  # we still need these ... for the early stop thing!!!
            #                logger.debug("Use 100% identity data")  # in test set, around 10% data will be with identities (lower than training set)
            #                id_train_df_idx = self.identity_idx

            self.train_mask[id_train_df_idx] = 1

            self.id_used_in_train = True

            for g in d.IDENTITY_COLUMNS:
                # column to keep recored what data is used in training, used in data_prepare module...
                self.train_df[g + "_in_train"] = 0.0
                # only the ones larger than 0.5 will ? how about negative example?
                self.train_df[g + "_in_train"].loc[id_train_df_idx] = self.train_df[
                    g
                ].loc[id_train_df_idx]

    def prepare_weight_for_subgroup_balance(self):
        """ to see how other people handle weights [this kernel](https://www.kaggle.com/thousandvoices/simple-lstm)
            sample_weights = np.ones(len(x_train), dtype=np.float32)
            # more weights for the ones with identities, more identities, more weights
            sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
            # the toxic ones, reverse identity (without identity)(average 4~8), so more weights on toxic one without identity
            sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
            # none toxic, non-toxic, with identity, more weight for this, so the weights are more or less balanced
            sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
            sample_weights /= sample_weights.mean()

        And we know the identies now, so we balance all the ones,
        for every subgroup, we calculate the related weight to balance
        """
        self.train_df = self.emb.train_df
        # self._get_identities()  # for known ones, just skip
        analyzer = d.TargetDistAnalyzer(self.train_df)
        self.target_analyzer = analyzer
        o = analyzer.get_distribution_overall()
        self.get_identities_for_training()
        # for subgroup, use 0.5 as the limit, continuous info not used... anyway, we try first
        gs = analyzer.get_distribution_subgroups()

        # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        balance_scheme_subgroups = BALANCE_SCHEME_SUBGROUPS
        # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        balance_scheme_across_subgroups = BALANCE_SCHEME_ACROSS_SUBGROUPS
        # balance_scheme_target_splits = 'target_bucket_same_for_target_splits' # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        # not work, because manual change will corrupt orignial information?
        balance_scheme_target_splits = BALANCE_SCHEME_TARGET_SPLITS
        balance_AUC = BALANCE_SCHEME_AUC

        # need a parameter for all pos v.s. neg., and for different
        def add_weight(balance_scheme_subgroups, balance_group=False):
            # target value, how do we balance?
            # (First we equalize them, then re-balance), just try different balance
            ones_weights = np.ones(len(self.train_df), dtype=np.float32)
            # sample_weights = ones_weights.copy()

            gs_weights_ratio = {}
            gs_weights = {}
            background_target_ratios = np.array([dstr[2] for dstr in o])

            if balance_scheme_subgroups == "target_bucket_same_for_subgroups":
                # compare with the background one, then change the weights to the same scale

                for g, v in gs.items():
                    gs_weights[g] = ones_weights.copy()  # initial, ones
                    # v is the distribution for ONE subgroup for 0~1 11 target types
                    gs_weights_ratio[g] = np.divide(
                        background_target_ratios, np.array(
                            [dstr[2] for dstr in v])
                    )

                    for target_split_idx, ratio in enumerate(gs_weights_ratio[g]):
                        # [3] is the index
                        split_idx_in_df = v[target_split_idx][3]
                        gs_weights[g][split_idx_in_df] *= ratio

            # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test

            if balance_scheme_across_subgroups == "more_for_low_score":
                subgroup_weights = {}
                subgroup_weights["homosexual_gay_or_lesbian"] = 4
                subgroup_weights["black"] = 3
                subgroup_weights["white"] = 3
                subgroup_weights["muslim"] = 2.5
                subgroup_weights["jewish"] = 4
                """
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
                """

                for g in subgroup_weights.keys():
                    subgroup_dist = gs[g]

                    for dstr in subgroup_dist:
                        split_idx_in_df = dstr[3]
                        # the ones with identities will be added
                        gs_weights[g][split_idx_in_df] *= subgroup_weights[g]

            # shape will be [sample nubmers , subgroups] as some sample might be in two groups
            weights_changer = np.transpose([v for v in gs_weights.values()])
            weights_changer_max = np.amax(weights_changer, axis=1)
            weights_changer_min = np.amin(weights_changer, axis=1)
            weights_changer_mean = np.mean(weights_changer, axis=1)
            weights_changer_merged = ones_weights.copy()
            weights_changer_merged[weights_changer_mean > 1] = weights_changer_max[
                weights_changer_mean > 1
            ]
            weights_changer_merged[weights_changer_mean < 1] = weights_changer_min[
                weights_changer_mean < 1
            ]

            sample_weights = weights_changer_merged

            if balance_AUC == "more_bp_sn":
                # self.train_df, contains all info
                benchmark_base = (
                    self.train_df[d.IDENTITY_COLUMNS +
                                  [d.TARGET_COLUMN, d.TEXT_COLUMN]]
                    .fillna(0)
                    .astype(np.bool)
                )
                # the idx happen to be the iloc value
                judge = d.BiasBenchmark(benchmark_base, threshold=0.5)
                # converted to binary in judge initailization function
                id_validate_df = judge.validate_df
                toxic_bool_col = id_validate_df[d.TARGET_COLUMN]
                contain_identity_bool_col = id_validate_df[d.IDENTITY_COLUMNS].any(
                    axis=1
                )

                weights_auc_balancer = ones_weights.copy() / 4
                # for subgroup postitive, will be 0.5 weight
                weights_auc_balancer[contain_identity_bool_col] += 1 / 4
                # BPSN, BP part (0.5 weights)
                weights_auc_balancer[toxic_bool_col & ~contain_identity_bool_col] += (
                    1 / 4
                )
                # still BPSN, SN part (0.75 weights)
                weights_auc_balancer[~toxic_bool_col & contain_identity_bool_col] += (
                    1 / 4
                )

                sample_weights = np.multiply(
                    sample_weights, weights_auc_balancer)

            wanted_split_ratios = None

            if balance_scheme_target_splits == "target_bucket_same_for_target_splits":
                wanted_split_ratios = [1 / len(background_target_ratios)] * len(
                    background_target_ratios
                )
            elif balance_scheme_target_splits == "target_bucket_extreme_positive":
                # 0 0.1 0.2 0.3 ... 1  # good
                wanted_split_ratios = [2, 2, 2, 2, 2, 10, 15, 20, 20, 15, 10]

            if wanted_split_ratios is not None:
                assert len(wanted_split_ratios) == len(
                    background_target_ratios)

                for target_split_idx, ratio in enumerate(background_target_ratios):
                    idx_for_split = o[target_split_idx][3]
                    # 1/len(b_t_r) is what we want
                    sample_weights[idx_for_split] *= (
                        wanted_split_ratios[target_split_idx] / ratio
                    )

            sample_weights /= sample_weights.mean()  # normalize

            return sample_weights

        weights = add_weight(balance_scheme_subgroups)

        return weights

    def prepare_train_labels(
            self,
            train_y_all,
            train_mask,
            custom_weights=False,
            with_aux=False,
            train_y_aux=None,
            sample_weights=None,
            fortify_subgroup=None,
    ):
        val_mask = np.invert(kernel.train_mask)  # this is whole train_mask

        if fortify_subgroup is not None:
            # only use the subgroup data
            train_mask = train_mask & (
                self.train_df[fortify_subgroup + "_in_train"] >= 0.5
            )

        train_X = kernel.train_X_all[train_mask]
        val_X = kernel.train_X_all[val_mask]

        if not custom_weights:
            if with_aux:
                return (
                    train_X,
                    val_X,
                    train_y_all[train_mask],
                    train_y_aux[train_mask],
                    train_y_all[val_mask],
                    train_y_aux[val_mask],
                )
            else:
                return train_X, val_X, train_y_all[train_mask], train_y_all[val_mask]
        else:
            # credit to https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution

            if sample_weights is None:
                raise RuntimeError(
                    "sample weights cannot be None if use custom_weights"
                )
            assert len(train_y_all) == len(sample_weights)

            if with_aux:
                return (
                    train_X,
                    val_X,
                    np.vstack([train_y_all, sample_weights]).T[train_mask],
                    train_y_aux[train_mask],
                    train_y_all[val_mask],
                    train_y_aux[val_mask],
                )
            else:
                return (
                    train_X,
                    val_X,
                    np.vstack([train_y_all, sample_weights]).T[train_mask],
                    train_y_all[val_mask],
                )

    def res_combine_pred_print_result(
        self, subgroup, y_pred, y_res_pred, idx_train, idx_val, detail=False
    ):
        id_df = copy.deepcopy(self.judge.validate_df)
        assert len(idx_train) + len(idx_val) == len(id_df[id_df[subgroup]])

        assert id_df.shape[0] == len(y_pred)

        model_name = "res_" + subgroup
        # there are comments mention two identity, so our way might not be good
        id_df[model_name] = y_pred
        # not iloc, both are index from the 1.8 Million data
        id_df.loc[idx_val, id_df.columns.get_loc(model_name)] = y_res_pred

        logger.debug(
            f"Res update for {subgroup}, {len(idx_val)} items predicted by res model"
        )

        self.calculate_metrics_and_print(
            validate_df_with_preds=id_df,
            model_name=model_name,
            detail=detail,
            file_for_print="metrics_log.txt",
        )

    def run_bias_auc_model(self):
        """
        need to prepare data, then train network to handle the bias thing
        we use data (identity(given value), comment text) as feature, to recalculate target, and reduce bias

        after build right model, then use predicted features to do the same prediction

        :return:
        """
        pass

    def load_identity_data_idx(self):
        if self.identity_idx is None:
            # to train the identity
            (
                self.train_X_identity,
                self.train_y_identity,
                self.identity_idx,
            ) = self.emb.get_identity_train_data_df_idx()

    def calculate_metrics_and_print(
        self,
        filename_for_print="metrics_log.txt",
        preds=None,
        threshold=0.5,
        validate_df_with_preds=None,
        model_name="lstm",
        detail=True,
        benchmark_base=None,
    ):
        file_for_print = open(filename_for_print, "w")

        self.emb.read_train_test_df(train_only=True)
        self.load_identity_data_idx()

        if benchmark_base is None:
            benchmark_base = self.train_df.loc[self.identity_idx]

        # if self.judge is None:  # no .... different threshold need to recalculate in the new judge
        # the idx happen to be the iloc value
        self.judge = d.BiasBenchmark(benchmark_base, threshold=threshold)
        self.id_validate_df = self.judge.validate_df

        if model_name == d.MODEL_NAME:
            if preds is not None:
                logger.debug(f"{model_name} result for {len(preds)} items:")

            if validate_df_with_preds is not None:
                logger.debug(
                    f"{model_name} result for {len(validate_df_with_preds)} items:"
                )

            if validate_df_with_preds is not None:
                (
                    value,
                    score_comp,
                    bias_metrics,
                    subgroup_distribution,
                    overall_distribution,
                ) = self.judge.calculate_benchmark(
                    validate_df=validate_df_with_preds, model_name=model_name
                )
            else:
                (
                    value,
                    score_comp,
                    bias_metrics,
                    subgroup_distribution,
                    overall_distribution,
                ) = self.judge.calculate_benchmark(preds)
        elif model_name.startswith("res"):
            logger.debug(
                f"{model_name} result for {len(validate_df_with_preds)} items in background"
            )
            (
                value,
                score_comp,
                bias_metrics,
                subgroup_distribution,
                overall_distribution,
            ) = self.judge.calculate_benchmark(
                validate_df=validate_df_with_preds, model_name=model_name
            )

        bias_metrics_df = bias_metrics.set_index("subgroup")

        # only the ones with identity is predicted
        pickle.dump(bias_metrics_df, open("bias_metrics", "wb"))
        # only the ones with identity is predicted
        pickle.dump(subgroup_distribution, open("subgroup_dist", "wb"))

        # bias_metrics_df = pickle.load(open("bias_metrics", 'rb'))  # only the ones with identity is predicted
        # subgroup_distribution = pickle.load(open("subgroup_dist", 'rb'))  # only the ones with identity is predicted

        if not detail:
            return

        print(
            f"bias metrics details (AUC, BPSN, BNSP): {score_comp}", file=file_for_print
        )
        print(
            f"final metric: {value} for threshold {threshold} applied to '{d.TARGET_COLUMN}' column, ",
            file=file_for_print,
        )
        # print("\n{}".format(bias_metrics[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']]), file=file_for_print)
        print(
            "\n{}".format(bias_metrics[["subgroup", "subgroup_auc"]]),
            file=file_for_print,
        )

        print("### subgroup auc", file=file_for_print)

        def d_str(t):
            num, mean, std = t

            return f"{num}, {mean:.4}, {std:.4}"

        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "subgroup_auc"
            s = "subgroup_size"
            auc = "{:.4} {}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3])
                + "\t"
                + d_str(d0[m][4])
                + "\t"
                + d_str(d0[m][5]),
                file=file_for_print,
            )

        print("### bpsn auc", file=file_for_print)

        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "bpsn_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3]),
                file=file_for_print,
            )

        print("### bnsp auc", file=file_for_print)

        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "bnsp_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3]),
                file=file_for_print,
            )

        print("### counts", file=file_for_print)
        # length thing

        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "subgroup_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + str(d0[m][0])
                + "\t"
                + str(d0[m][1]),
                file=file_for_print,
            )

        print("### overall", file=file_for_print)
        g = "overall"
        m = d.OVERALL_AUC
        s = "subgroup_size"
        auc = "{0:.4} {1}".format(
            overall_distribution[m], overall_distribution[s])
        dist = overall_distribution["distribution"]
        print(
            f"{g:5.5} {auc}\tneg_tgt_pred_dis:{d_str(dist[2])}\tpos_tgt_pred_dis:{d_str(dist[3])}\noverall_pos_neg_cnt:\t{dist[0]}",
            file=file_for_print,
        )
        print(
            f"{g:5.5} {auc}\tneg_tgt_act_dis:{d_str(dist[4])}\tpos_tgt_act_dis:{d_str(dist[5])}",
            file=file_for_print,
        )

        file_for_print.close()
        file_for_print = open(filename_for_print, "r")
        print(file_for_print.read())

        return value  # final value


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument(
    "--train_steps", default=2, type=int, help="number of training steps"
)
parser.add_argument("--learning_rate", default=0.001,
                    type=float, help="learing rate")


# FOCAL_LOSS_GAMMA = 2.
# ALPHA = 0.666
# FOCAL_LOSS_GAMMA = 0.
# ALPHA = 0.7(0.9121)  # 0.9(0.9122), 0.8(0.9123...) (no difference with no focal loss)
# GAMMA works better 2. with BS 1024
# GAMMA works better 1.5 with BS 512

# _debug_train_data = None

CONVERT_DATA = False
CONVERT_DATA_Y_NOT_BINARY = "convert_data_y_not_binary"
# given the pickle of numpy train data
CONVERT_TRAIN_DATA = "convert_train_data"
# given the pickle of numpy train data
CONVERT_ADDITIONAL_NONTOXIC_DATA = "CONVERT_ADDITIONAL_NONTOXIC_DATA"

EXCLUDE_IDENTITY_IN_TRAIN = True
TRAIN_DATA_EXCLUDE_IDENDITY_ONES = "TRAIN_DATA_EXCLUDE_IDENDITY_ONES"
DATA_ACTION_NO_NEED_LOAD_EMB_M = "DATA_ACTION_NO_NEED_LOAD_EMB_M"

NEG_RATIO = 1 - 0.05897253769515213


def main(argv):
    # args = parser.parse_args(argv[1:])

    global kernel
    logger.info("Will start load data")
    # kernel = KaggleKernel(action="convert_train_data")
    action = None

    if CONVERT_DATA:
        # action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
        # action = CONVERT_DATA_Y_NOT_BINARY
        # action = CONVERT_TRAIN_DATA  # given the pickle of numpy train data
        # given the pickle of numpy train data
        action = CONVERT_ADDITIONAL_NONTOXIC_DATA
    else:
        if (
            EXCLUDE_IDENTITY_IN_TRAIN and not IDENTITY_RUN
        ):  # for identity, need to predict for all train data
            action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
    # if not (RESTART_TRAIN or RESTART_TRAIN_ID or RESTART_TRAIN_RES):
    #    action = DATA_ACTION_NO_NEED_LOAD_EMB_M  # loading model from h5 file, no need load emb matrix (save memory)

    if FINAL_SUBMIT:
        kernel = KaggleKernel(action=None)
    else:
        kernel = KaggleKernel(action=action)
        logger.debug(action)
        # TODO put to my kernel class
    kernel.load_identity_data_idx()  # so only predict part of the data

    logger.info("load data done")

    # pred = pickle.load(open('predicts', 'rb'))
    prefix = kernel.emb.BIN_FOLDER

    if TARGET_RUN == "res":
        # if not os.path.isfile('predicts'):
        preds = pickle.load(open("predicts", "rb"))
        kernel.calculate_metrics_and_print(preds)
        # improve the model with another data input,

    elif TARGET_RUN == "lstm":
        predict_only = PRD_ONLY

        if not TARGET_RUN_READ_RESULT:
            if ANA_RESULT:
                # preds = pickle.load(open('predicts', 'rb'))
                # sample_weight = kernel.prepare_weight_for_subgroup_balance()
                pass
            else:
                sample_weights = kernel.prepare_weight_for_subgroup_balance()
                # sample_weights_train = sample_weights[kernel.train_mask]
                # pickle.dump(val_mask, open("val_mask", 'wb'))  # only the ones with identity is predicted
                val_mask = np.invert(kernel.train_mask)
                # val..., no need to weight changing
                sample_weights[val_mask] = 1.0

                train_X = None
                val_X = None
                train_y_aux = None
                train_y = None
                val_y_aux = None

                if NO_AUX:
                    train_X, val_X, train_y, val_y = kernel.prepare_train_labels(
                        kernel.train_y_all,
                        kernel.train_mask,
                        custom_weights=True,
                        with_aux=False,
                        train_y_aux=None,
                        sample_weights=sample_weights,
                    )
                else:
                    (
                        train_X,
                        val_X,
                        train_y,
                        train_y_aux,
                        val_y,
                        val_y_aux,
                    ) = kernel.prepare_train_labels(
                        kernel.train_y_all,
                        kernel.train_mask,
                        custom_weights=True,
                        with_aux=True,
                        train_y_aux=kernel.train_y_aux_all,
                        sample_weights=sample_weights,
                        fortify_subgroup=None,
                    )
                # sample_weights_train = None  # no need to use in fit, will cooperate to the custom loss function
                logger.debug(f"train with {len(train_X)} data entries")
                logger.debug(f"prefix: {prefix}")
                preds = kernel.run_lstm_model(
                    predict_ones_with_identity=True,
                    final_train=FINAL_SUBMIT,
                    params={
                        "prefix": prefix,
                        # will retrain every time if True,restore will report sensitivity problem now
                        "re-start-train": RESTART_TRAIN,
                        "predict-only": predict_only,
                        "starter_lr": STARTER_LEARNING_RATE,
                        # 'sample_weights': sample_weights_train,
                        # train data with identities
                        "train_data": (train_X[:10000], train_y[:10000,:]),
                        "train_y_aux": train_y_aux,
                        "val_y_aux": val_y_aux,
                        # train data with identities
                        "val_data": (val_X, val_y),
                        "patience": 2,
                        "binary": True,
                        # 'fortify_subgroup': 'black',
                    },
                    val_mask=val_mask,
                )  # only the val_mask ones is predicted TODO modify val set, to resemble test set
                # if not FINAL_SUBMIT:  # we still need to check this ... for evalutate our model
        else:
            # ans is in self.train_df.loc[val_mask, 'lstm'] = preds
            preds, val_mask = pickle.load(open("pred_val_mask", "rb"))

        # else:
        #    preds = pickle.load(open('predicts', 'rb'))
        # df[df.white & (df.target_orig<0.5) & (df.lstm > 0.5)][['comment_text','lstm','target_orig']].head()
        # kernel.evaluate_model_and_print(preds, 0.55)

        # todo later we could split train/test, to see overfit thing, preds here are all ones with identity, need to
        #  split out the ones is in the training set

        # check if binarify will make difference -> yes, the result is worse
        # pred_target = np.where(preds[0] >= 0.5, True, False)
        # value, bias_metrics = kernel.evaluate_model(pred_target)
        # logger.info(value)
        # logger.info(f"\n{bias_metrics}")

    return


# or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
BALANCE_SCHEME_SUBGROUPS = "target_bucket_same_for_subgroups"
# or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
BALANCE_SCHEME_ACROSS_SUBGROUPS = "more_for_low_score"
BALANCE_SCHEME_AUC = "more_bp_sn"
# balance_scheme_target_splits = 'target_bucket_same_for_target_splits' # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
# not work, because manual change will corrupt orignial information?
BALANCE_SCHEME_TARGET_SPLITS = "no_target_bucket_extreme_positive"

WEIGHT_TO_Y = True


ANA_RESULT = False

if os.path.isfile(".ana_result"):
    ANA_RESULT = True
    RESTART_TRAIN = False

    IDENTITY_RUN = False
    TARGET_RUN = "lstm"
    PRD_ONLY = True
    RESTART_TRAIN_RES = False

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logger = utils.logger
    logger.info(
        f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}\n{BALANCE_SCHEME_SUBGROUPS} {BALANCE_SCHEME_ACROSS_SUBGROUPS} {BALANCE_SCHEME_TARGET_SPLITS} {BALANCE_SCHEME_AUC}"
    )
    main([1])
    # tf.compat.v1.app.run(main)
    logger.info(
        f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}\n{BALANCE_SCHEME_SUBGROUPS} {BALANCE_SCHEME_ACROSS_SUBGROUPS} {BALANCE_SCHEME_TARGET_SPLITS} {BALANCE_SCHEME_AUC}"
    )
