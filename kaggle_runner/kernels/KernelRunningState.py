from enum import Enum


class KernelRunningState(Enum):
    INIT_DONE = 1
    PREPARE_DATA_DONE = 2
    TRAINING_DONE = 3
    EVL_DEV_DONE = 4
    SAVE_SUBMISSION_DONE = 5

    @staticmethod
    def string(state_int_value):
        names = [
            "INIT_DONE",
            "PREPARE_DATA_DONE",
            "TRAINING_DONE",
            "EVL_DEV_DONE",
            "SAVE_SUBMISSION_DONE",
        ]
        if state_int_value is not None:
            return names[state_int_value]
        else:
            return ""
