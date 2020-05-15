import tensorflow.keras.backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    Layer,
    PReLU,
    SpatialDropout1D,
    add,
    concatenate,
)

from kaggle_runner.kernels.KernelRunningState import KernelRunningState
from kaggle_runner.utils.kernel_utils import dump_obj, get_obj_or_dump, logger


class AttentionRaffel(Layer):
    def __init__(
        self,
        step_dim,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs,
    ):
        """
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        :param step_dim: feature vector length
        :param W_regularizer:
        :param b_regularizer:
        :param W_constraint:
        :param b_constraint:
        :param bias:
        :param kwargs:
        """
        super(AttentionRaffel, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = "glorot_uniform"

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = {
            "step_dim": self.step_dim,
            "bias": self.bias,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
        }
        base_config = super(AttentionRaffel, self).get_config()
        if "cell" in base_config:
            del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight(
            "{}_W".format(self.name),
            (int(input_shape[-1]),),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight(
                "{}_b".format(self.name),
                (int(input_shape[1]),),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # more like the alignment model, which scores how the inputs around
        # position j and the output at position i match
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.dot(
                K.reshape(x, (-1, features_dim)
                          ), K.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim),
        )

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)  # activation

        # softmax
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may
        # be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        # context vector c_i (or for this, only one c_i)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

    def set_metrics(self):
        """
        set_metrics for model training

        :return: None
        """
        pass

    def set_result_analyzer(self):
        pass

    def pre_prepare_data_hook(self):
        pass

    def after_prepare_data_hook(self):
        pass

    def prepare_train_dev_data(self):
        pass

    def prepare_test_data(self):
        pass

    def predict_on_test(self):
        pass

    def dump_state(self, exec_flag=False, force=True):
        logger.debug(f"state {self._stage}")
        if exec_flag:
            logger.debug(f"dumping state {self._stage}")
            dump_obj(self, f"run_state_{self._stage}.pkl", force=force)
            # dump_obj(self, 'run_state.pkl', force=True)  # too large

    def run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
        force_dump=True,
    ):
        """

        :param start_stage: if set, will overwrite the stage
        :param end_stage:
        :param dump_flag:
        :return:
        """
        self.continue_run(
            start_stage=start_stage,
            end_stage=end_stage,
            dump_flag=dump_flag,
            force_dump=force_dump,
        )

    def continue_run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
        force_dump=True,
    ):
        if start_stage is not None:
            assert start_stage.value < end_stage.value
            self._stage = start_stage

        if self._stage.value < KernelRunningState.PREPARE_DATA_DONE.value:
            self.pre_prepare_data_hook()
            self.prepare_train_dev_data()
            self.after_prepare_data_hook()

            self._stage = KernelRunningState.PREPARE_DATA_DONE
            self.dump_state(exec_flag=dump_flag, force=force_dump)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.TRAINING_DONE.value:
            self.pre_train()
            self.build_and_set_model()
            self.train_model()
            self.after_train()

            self.save_model()

            self._stage = KernelRunningState.TRAINING_DONE
            self.dump_state(exec_flag=dump_flag, force=force_dump)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.EVL_DEV_DONE.value:
            self.set_result_analyzer()

            self._stage = KernelRunningState.EVL_DEV_DONE
            self.dump_state(exec_flag=False, force=force_dump)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.SAVE_SUBMISSION_DONE.value:
            self.pre_test()
            self.prepare_test_data()
            self.predict_on_test()
            self.after_test()

            self._stage = KernelRunningState.SAVE_SUBMISSION_DONE
            self.dump_state(exec_flag=False, force=force_dump)
            if self._stage.value >= end_stage.value:
                return

    @classmethod
    def _load_state(cls, stage=None, file_name="run_state.pkl"):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        if stage is not None:
            file_name = f"run_state_{stage}.pkl"
        logger.debug(f"restore from {file_name}")
        return get_obj_or_dump(filename=file_name)

    def load_state_data_only(self, file_name="run_state.pkl"):
        pass

    @classmethod
    def load_state_continue_run(cls, file_name="run_state.pkl"):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        self = cls._load_state(file_name=file_name)
        self.continue_run()

    def pre_train(self):
        pass

    def after_train(self):
        pass

    def pre_test(self):
        pass

    def after_test(self):
        pass
