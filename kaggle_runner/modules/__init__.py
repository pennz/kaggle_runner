import tensorflow as tf

from lstm import LSTM_UNITS, DENSE_HIDDEN_UNITS


def reinitLayers(model):
    # reinitLayers refer to https://github.com/keras-team/keras/issues/341
    session = K.get_session()
    for layer in model.layers:
        # if isinstance(layer, keras.engine.topology.Container):
        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        if layer.trainable == False:
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(
                v_arg, "initializer"
            ):  # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, "initializer")
                initializer_method.run(session=session)
                print("reinitializing layer {}.{}".format(layer.name, v))


def identity_model(features, labels, mode, params):
    # words = Input(shape=(None,))
    embedding_matrix = params["embedding_matrix"]
    identity_out_num = params["identity_out_num"]

    words = features  # input layer, so features need to be tensor!!!

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(
        words
    )
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])

    # Output Layer
    result = Dense(identity_out_num, activation="sigmoid")(hidden)

    # Implement training, evaluation, and prediction
    # Compute predictions.
    # predicted_classes = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # predictions = {
        #    'class_ids': predicted_classes[:, tf.newaxis],
        #    'probabilities': tf.nn.softmax(logits),
        #    'logits': logits,
        # }
        return tf.estimator.EstimatorSpec(mode, predictions=result.out)

    # Compute loss.
    loss = tf.keras.losses.binary_crossentropy(
        labels, result
    )  # todo put it together, fine?

    # Compute evaluation metrics.
    # m = tf.keras.metrics.SparseCategoricalAccuracy()
    # m.update_state(labels, logits)
    # accuracy = m.result()
    # metrics = {'accuracy': accuracy}
    # tf.compat.v1.summary.scalar('accuracy', accuracy)
    binary_accuracy = tf.keras.metrics.binary_accuracy(labels, result)

    metrics = {"accuracy": binary_accuracy}
    tf.compat.v1.summary.scalar("accuracy", binary_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op. # will be called by the estimator
    assert mode == tf.estimator.ModeKeys.TRAIN

    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params["learning_rate"],
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=params["decay_steps"],
            staircase=True,
            decay_rate=0.5,
        )
    )
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)