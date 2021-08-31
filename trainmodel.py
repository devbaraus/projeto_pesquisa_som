# %%
import sys
from tensorflow import keras
from deep_audio import Directory, Process, Terminal, Model, JSON
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


model = globals()['build_'+method](**param_grid)

higher_score = 12340

filename_holder = 'models/' + Directory.model_filename(method, language, library, normalization,
                                                       higher_score, augmentation=augment, json=False)

model_save_filename = filename_holder + 'weight.h5'

# DECIDE QUANDO PARAR
earlystopping_cb = keras.callbacks.EarlyStopping(
    patience=300, restore_best_weights=True)

# SALVA OS PESOS
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)


history = model.fit(X_train, y_train, epochs=param_grid['epochs'],
                    batch_size=batch_size, validation_data=(X_valid, y_valid))


# GERA O GRAFICO DE ACURÁCIA
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(filename_holder + 'graph_accuracy.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(filename_holder + 'graph_loss.png')
plt.close()

JSON.create_json_file(
    file=filename_holder + 'model.json',
    data=globals()['build_' + method]().to_json()
)

higher_score = model.evaluate(X_test, y_test, batch_size=batch_size)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_model(
    filename_holder + 'info.json',
    model=param_grid,
    language=language,
    method=method,
    normalization=normalization,
    sampling_rate=sampling_rate,
    augmentation=augment,
    shape=X_train.shape,
    seed=random_state,
    library=library,
    sizes=[len(X_train), len(X_valid), len(X_test)],
    score_train=history.history['accuracy'][1],
    score_test=higher_score,
    extra={
        'epochs': epochs,
        'batch_size': batch_size,
        'micro_f1': f1_score(y_test, model.predict_classes(X_test), average='micro'),
        'macro_f1': f1_score(y_test, model.predict_classes(X_test), average='macro')
    }
)


Directory.rename_directory(filename_holder,
                           'models/' + Directory.model_filename(method, language, library, normalization,
                                                                higher_score, augmentation=augment, json=False))
