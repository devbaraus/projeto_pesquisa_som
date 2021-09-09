# %%
from pickle import dump
import sys
from tensorflow import keras
from tensorflow.keras import callbacks
from deep_audio import Directory, Process, Terminal, Model, JSON
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# %%
args = Terminal.get_args()

language = args['language']
method = args['method']
library = args['representation']
people = args['people']
segments = args['segments']
normalization = args['normalization']
flat = args['flat']
augment = args['augmentation']
sampling_rate = 24000
random_state = 42

epochs = 2000
batch_size = 128
# %%
file_path = Directory.processed_filename(
    language, library, sampling_rate, people, segments, augment)
# %%
X_train, y_train, mapping = Process.selection(
    file_path, valid_size=0, test_size=0, mapping=True)


param_grid = {}

# %%
if normalization == 'minmax':
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

elif normalization == 'standard':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

if method in ['perceptron', 'lstm']:
    if len(X_train.shape) > 3:
        X_train = X_train[:, :, :, 0]


if method in ['lstm']:

    param_grid = {
        'lstm': 96,
        'dropout1': 0.2,
        'dense1': 64,
        'dense2': 32,
        'dropout2': 0.2,
        'dense3': 24,
        'dropout3': 0.4,
        # 'epochs': 2000,
    }

elif method in ['cnn']:
    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        # X_valid = X_valid[..., np.newaxis]

    param_grid = {
        'resizing': (32, 32),
        'conv2d1': 32,
        'conv2d2': 64,
        'dropout1': 0.25,
        'dropout2': 0.5,
        'dense': 128,
        # 'epochs': 2000,
    }

elif method in ['perceptron']:
    param_grid = {
        'dense1': 512,
        'dense2': 256,
        'dense3': 128,
        # 'epochs': 2000,
    }


def build_cnn(resizing=(32, 32), conv2d1=32, conv2d2=64, dropout1=0.25, dropout2=0.5, dense=128, learning_rate=0.0001):
    from tensorflow.keras.layers.experimental import preprocessing

    model = keras.Sequential()

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    outputs = len(set(y_train))

    model.add(keras.layers.Input(shape=input_shape))
    model.add(preprocessing.Resizing(resizing[0], resizing[1]))
    model.add(keras.layers.Conv2D(conv2d1, 3, activation='relu'))
    model.add(keras.layers.Conv2D(conv2d2, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(dropout1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(keras.layers.Dense(outputs,  activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_perceptron(dense1=512, dense2=256, dense3=128, learning_rate=0.0001):
    model = keras.Sequential()

    outputs = len(set(y_train))

    if not flat:
        model.add(keras.layers.Flatten(
            input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(keras.layers.Dense(dense1, activation='relu'))
    model.add(keras.layers.Dense(dense2, activation='relu'))
    model.add(keras.layers.Dense(dense3, activation='relu'))
    model.add(keras.layers.Dense(outputs, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_lstm(lstm=64, dropout1=0.2, dense1=64, dense2=32, dropout2=0.4, dense3=24, dropout3=0.4, learning_rate=0.0001):
    from tensorflow import keras

    input_shape = (X_train.shape[1], X_train.shape[2])
    outputs = len(set(y_train))

    model = keras.Sequential()
    model.add(keras.layers.LSTM(lstm, input_shape=input_shape))
    model.add(keras.layers.Dropout(dropout1))
    model.add(keras.layers.Dense(dense1, activation='relu'))
    model.add(keras.layers.Dense(dense2, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(keras.layers.Dense(dense3, activation='relu'))
    model.add(keras.layers.Dropout(dropout3))
    model.add(keras.layers.Dense(outputs, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = globals()['build_'+method](**param_grid)

filename_holder = Directory.model_filename(
    method=method,
    language=language,
    library=library,
    normalization=normalization,
    augmentation=augment,
    json=False,
    models=True
)

model_save_filename = filename_holder + 'weight.h5'

# SALVA OS PESOS
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="accuracy", save_best_only=True
)

# history = model.fit(X_train, y_train, epochs=epochs,
#                     batch_size=batch_size, callbacks=[mdlcheckpoint_cb])

# SALVA ESTRUTURA DO MODELO
JSON.create_json_file(
    file=filename_holder + 'model.json',
    data=globals()['build_' + method]().to_json()
)

dump(scaler, open(filename_holder + 'scaler.pkl', 'wb'))

# SALVA OS PARAMETROS
Model.dump_model(
    filename_holder + 'info.json',
    params=param_grid,
    language=language,
    method=method,
    normalization=normalization,
    sampling_rate=sampling_rate,
    augmentation=augment,
    shape=X_train.shape,
    seed=random_state,
    library=library,
    extra={
        'epochs': epochs,
        'batch_size': batch_size,
        'mapping': mapping
    }
)
