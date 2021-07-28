# %%
from time import time
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from tensorflow.python.keras.layers import pooling
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from deep_audio import Directory, JSON, Process, Terminal, Model
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
# %%
args = Terminal.get_args()

language = args['language'] or 'portuguese'
library = args['representation'] or 'stft'
people = args['people'] or None
segments = args['segments'] or None
sampling_rate = 24000
random_state = 42
normalization = args['normalization'] or 'nonorm'
flat = args['flat']

epochs = 500
batch_size = 128
# %%
global X_train, X_valid, X_test, y_train, y_valid, y_test

file_path = Directory.processed_filename(
    language, library, sampling_rate, people, segments)
# %%
if language == 'mixed' and library == 'mixed':
    first_folder = Directory.processed_filename(
        'portuguese', 'psf', sampling_rate, None, None)
    second_folder = Directory.processed_filename(
        'portuguese', 'melbanks', sampling_rate, None, None)
    third_folder = Directory.processed_filename(
        'english', 'psf', sampling_rate, people, segments)
    fourth_folder = Directory.processed_filename(
        'english', 'melbanks', sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection(
        first_folder, second_folder, third_folder, fourth_folder,
        lm_validation=False,
        lm_test=False,
        rm_validation=True,
        rm_test=True
    )
elif language == 'mixed':
    portuguese_folder = Directory.processed_filename(
        'portuguese', library, sampling_rate, people, segments)
    english_folder = Directory.processed_filename(
        'english', library, sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_language(
        portuguese_folder=portuguese_folder,
        english_folder=english_folder,
        flat=False
    )
elif library == 'mixed':
    first_folder = Directory.processed_filename(
        language, 'psf', sampling_rate, people, segments)
    second_folder = Directory.processed_filename(
        language, 'melbanks', sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_representation(
        first_folder,
        second_folder,
        test=True)
else:
    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.selection(
        file_path, flat=flat)

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

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(X_train.shape)


def build_model():
    # build the network architecture

    model = keras.Sequential()

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # if library != 'mixed':
    #     model.add(keras.layers.Flatten(
    #         input_shape=(X_train.shape[1], X_train.shape[2])))

    # AISHELLL 1
    # model.add(keras.layers.Input(shape=input_shape))
    # model.add(keras.layers.Conv2D(
    #     kernel_size=(5, 5), strides=(2, 2), filters=64))
    # model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.SimpleRNN(1024))
    # model.add(keras.layers.SimpleRNN(1024))
    # model.add(keras.layers.SimpleRNN(1024))
    # model.add(keras.layers.average())
    # model.add(keras.layers.Dense(len(set(y_train)), activation='softmax'))

    # VALERIO VELARDO
    # model.add(keras.layers.Conv2D(
    #     64, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Conv2D(
    #     48, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Conv2D(
    #     32, (2, 2), activation='relu', input_shape=input_shape))
    # model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Flatten())

    # model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))

    # model.add(keras.layers.Dense(len(set(y_train)), activation='softmax'))

    # DANIEL NOTEBOOK CNN
    # model.add(keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
    #                               input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(keras.layers.GRU(20, return_sequences=True))
    # model.add(keras.layers.GRU(20, return_sequences=True))
    # model.add(keras.layers.TimeDistributed(
    #     keras.layers.Dense(len(set(y_train)))))

    # BIOMETRIA DE LA VOZ
    model.add(keras.layers.Input(shape=input_shape))
    model.add(preprocessing.Resizing(32, 32))
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(set(y_train)),  activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


kc = KerasClassifier(build_fn=build_model,
                     epochs=epochs, batch_size=batch_size, verbose=1)

param_grid = {}

model = GridSearchCV(
    estimator=kc, param_grid=param_grid, n_jobs=-1, cv=5)


model.fit(X_train, y_train)

best_params = model.best_params_

score_test = model.score(X_test, y_test)

score_train = model.score(X_train, y_train)

y_hat = model.predict(X_test)

filename_model = Directory.model_filename(
    'cnn', language, library, normalization, score_test, json=False)+'model.json'

JSON.create_json_file(
    file=filename_model,
    data=build_model().to_json()
)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_grid(
    Directory.model_filename(
        'cnn', language, library, normalization, score_test),
    model=model,
    language=language,
    method='CNN',
    sampling_rate=sampling_rate,
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
