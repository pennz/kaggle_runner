from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import (Callback, CSVLogger, ModelCheckpoint,
                                        ReduceLROnPlateau)

from kaggle_runner import logger


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)

            if y_pred.size > self.y_val.size:
                y_pred = y_pred[:,0]
            score = roc_auc_score(self.y_val, y_pred)
            print(
                "\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            logger.debug(
                "\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


def ReduceLROnPlateauLogCBs(validation_data):
    cb = []

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.3, patience=3,
                                       verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=1, min_lr=0.000001)
    cb.append(reduceLROnPlat)
    log = CSVLogger('log.csv')
    cb.append(log)

    RocAuc = RocAucEvaluation(validation_data=validation_data, interval=1)
    cb.append(RocAuc)
    # ckpt = ModelCheckpoint("toxic.hdf5", save_best_only=True, verbose=1) bert get_config not implemented, cannot save
    # cb.append(ckpt)

    return cb
