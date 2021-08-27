# %%
from time import time
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from deep_audio import Directory, Process, Terminal, Model
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
# %%
args = Terminal.get_args()

language = args['language']
library = args['representation']
people = args['people']
segments = args['segments']
sampling_rate = 24000
random_state = 42

# language = 'mixed'
# library = 'psf'
# people = None
# segments = None
# sampling_rate = 24000
# random_state = 42
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
        file_path, flat=False, squeeze=False)

# %%


def build_model():
    # build the network architecture
    input_shape = (X_train.shape[1], X_train.shape[2])
    outputs = len(set(y_train))

    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=input_shape))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(outputs, activation='softmax'))
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


kc = KerasClassifier(build_fn=build_model,
                     epochs=500, batch_size=64, verbose=1)

param_grid = {}

model = GridSearchCV(
    estimator=kc, param_grid=param_grid, n_jobs=-1, cv=2)


model.fit(X_train, y_train)

best_params = model.best_params_

score_test = model.score(X_test, y_test)

score_train = model.score(X_train, y_train)

y_hat = model.predict(X_test)

filename_ps = Directory.verify_people_segments(
    people=people, segments=segments)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_grid(
    f'{language}/models/lstm/{library}/{filename_ps}{Process.pad_accuracy(score_test)}_{abs(time())}/info.json',
    model=model,
    language=language,
    method='LSTM',
    sampling_rate=sampling_rate,
    seed=random_state,
    library=library,
    sizes=[len(X_train), len(X_valid), len(X_test)],
    score_train=score_train,
    score_test=score_test,


)
