# %%
import sys
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from deep_audio import Directory, Process, Terminal, Model, JSON
import numpy as np
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
X_train, X_valid, X_test, y_train, y_valid, y_test = Process.selection(
    file_path, flat=flat)

param_grid = {}

# %%
if normalization == 'minmax':
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

elif normalization == 'standard':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

if method in ['perceptron', 'lstm']:
    if len(X_train.shape) > 3:
        X_train = X_train[:, :, :, 0]
        X_valid = X_valid[:, :, :, 0]
        X_test = X_test[:, :, :, 0]


if method in ['lstm']:
    # param_grid = {
    #     'lstm': [32, 64, 96],
    #     'dropout1': [0, 0.2, 0.4],
    #     # 'dense1': [32, 64],
    #     'dense2': [32, 64],
    #     'dropout2': [0, 0.2, 0.4],
    #     # 'dense3': [24, 32],
    #     # 'dropout3': [0, 0.2, 0.4],
    #     # 'learning_rate': [0.001, 0.0001],
    #     'epochs': [2000],
    # }

    param_grid = {
        'lstm': [96],
        'dropout1': [0.2],
        'dense1': [64],
        'dense2': [32],
        'dropout2': [0.2],
        'dense3': [24],
        'dropout3': [0.4],
        'epochs': [2000],
    }

elif method in ['cnn']:
    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_valid = X_valid[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    # param_grid = {
    #     'resizing': [(32, 32), (64, 64)],
    #     # 'conv2d1': [32, 64],
    #     'conv2d2': [32, 64],
    #     'dropout1': [0, 0.25, 0.5],
    #     # 'dropout2': [0, 0.25, 0.5],
    #     'dense': [128, 64],
    #     # 'learning_rate': [0.001, 0.0001],
    #     'epochs': [2000],
    # }

    param_grid = {
        'resizing': [(32, 32)],
        'conv2d1': [32],
        'conv2d2': [64],
        'dropout1': [0.25],
        'dropout2': [0.5],
        'dense': [128],
        'epochs': [2000],
    }


def build_cnn(resizing=(32, 32), conv2d1=32, conv2d2=64, dropout1=0.25, dropout2=0.5, dense=128, learning_rate=0.0001):
    from tensorflow import keras
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


def build_perceptron():
    return Model.build_mlp(X_train, y_train)


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


kc = KerasClassifier(
    build_fn=globals()['build_' + method],
    epochs=epochs,
    batch_size=batch_size,
)


model = GridSearchCV(
    estimator=kc, param_grid=param_grid, n_jobs=-1, cv=5, verbose=5)


model.fit(X_train, y_train)

best_params = model.best_params_

score_test = model.score(X_test, y_test)

score_train = model.score(X_train, y_train)

y_hat = model.predict(X_test)

filename_model = Directory.model_filename(
    method, language, library, normalization, score_test, augmentation=augment, json=False)+'model.json'

JSON.create_json_file(
    file=filename_model,
    data=globals()['build_' + method]().to_json()
)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_grid(
    Directory.model_filename(
        method, language, library, normalization, score_test, augmentation=augment),
    model=model,
    language=language,
    method=method,
    normalization=normalization,
    sampling_rate=sampling_rate,
    augmentation=augment,
    shape=X_train.shape,
    seed=random_state,
    library=library,
    sizes=[len(X_train), len(X_valid), len(X_test)],
    score_train=score_train,
    score_test=score_test,
    model_file=filename_model,
    extra={
        'epochs': epochs,
        'batch_size': batch_size
    }
)
