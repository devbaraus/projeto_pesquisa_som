import numpy as np
import sys


def build_lstm(documents, classes, flat=False):
    from tensorflow import keras

    input_shape = (documents.shape[1], documents.shape[2])
    outputs = len(set(classes))

    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=input_shape))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(outputs, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_cnn(documents, classes, flat=False):
    from tensorflow import keras
    from tensorflow.keras.layers.experimental import preprocessing

    model = keras.Sequential()

    input_shape = (documents.shape[1], documents.shape[2], documents.shape[3])
    outputs = len(set(classes))

    model.add(keras.layers.Input(shape=input_shape))
    model.add(preprocessing.Resizing(32, 32))
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(outputs,  activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_mlp(documents, classes, flat=False):
    from tensorflow import keras

    model = keras.Sequential()

    outputs = len(set(classes))

    if not flat:
        model.add(keras.layers.Flatten(
            input_shape=(documents.shape[1], documents.shape[2])))

    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(outputs, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model_json_file(file, data):
    from deep_audio import Directory

    directory = '/'.join(file.split('/')[:-1])

    directory.create_directory(directory)

    with open(file, "w") as json_file:
        json_file.write(data)


def dump_grid(file, model, language, method, seed, library, sizes, score_train, score_test, sampling_rate, normalization, shape, augmentation, score_valid=None,
              model_file=None, extra={}):
    from time import time
    from deep_audio import JSON

    dump_info = {
        'method': method,
        'language': language,
        'normalization': normalization,
        'seed': seed,
        'augmentation': augmentation,
        'library': library,
        'sample_rate': sampling_rate,
        'shape': shape,
        'sizes': sizes,
        'score_train': score_train,
        'score_test': score_test,
        'timestamp': time(),
        'params': model.best_params_,
        'cv_results': model.cv_results_,
        **extra
    }

    if score_valid:
        dump_info['score_valid'] = score_valid

    if model:
        dump_info['model_file'] = model_file

    JSON.create_json_file(file, dump_info, cls=JSON.NumpyEncoder)

    return


def dump_model(file, params, language, method, seed, library, sizes, score_train, score_test, sampling_rate, normalization, shape, augmentation, extra={}):
    from time import time
    from deep_audio import JSON

    dump_info = {
        'method': method,
        'language': language,
        'normalization': normalization,
        'seed': seed,
        'augmentation': augmentation,
        'library': library,
        'sample_rate': sampling_rate,
        'shape': shape,
        'sizes': sizes,
        'score_train': score_train,
        'score_test': score_test,
        'timestamp': time(),
        'params': params,
        **extra
    }

    JSON.create_json_file(file, dump_info, cls=JSON.NumpyEncoder)

    return


def load_processed_data(path, inputs_fieldname='mfcc'):
    import json
    import numpy as np

    with open(path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data[inputs_fieldname])
    targets = np.array(data['labels'])
    mapping = data['mapping']

    return inputs, targets, mapping
