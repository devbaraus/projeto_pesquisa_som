import json
import numpy as np
from tensorflow import keras
from deep_audio import Terminal, Directory, Audio
from pickle import load
from leaf_audio.frontend import MelFilterbanks
from tensorflow import newaxis
import librosa

args = Terminal.get_args()

language = 'portuguese'
method = 'cnn'
library = 'melbanks'
people = args['people']
segments = args['segments']
normalization = args['normalization']
augment = args['augmentation']
sampling_rate = 24000
random_state = 42

learning_rate = 0.0001
epochs = 2000
batch_size = 128


filename_holder = Directory.model_filename(
    method=method,
    language=language,
    library=library,
    normalization=normalization,
    augmentation=augment,
    json=False,
    models=True
)

info = json.load(open(filename_holder+'info.json', 'r'))
scaler = load(open(filename_holder + 'scaler.pkl', 'rb'))


def build_cnn(resizing=(32, 32), conv2d1=32, conv2d2=64, dropout1=0.25, dropout2=0.5, dense=128, learning_rate=0.0001):
    from tensorflow.keras.layers.experimental import preprocessing

    outputs = len(info['mapping'])

    input_shape = (info['shape'][1], info['shape'][2], 1)

    model = keras.Sequential()
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


model = build_cnn()

model.load_weights(filename_holder + 'weight.h5')

signal, rate = librosa.load(args['inferencia'], sr=sampling_rate)

# signal = Audio.trim(signal)

segment_time = 5
signal = signal[:len(signal) - len(signal) % (rate * segment_time)]

segments = len(signal) // (rate * segment_time)

mfcc_audios = []

for i in range(segments):
    sample = Audio.segment(signal, rate, seconds=segment_time, window=i)

    sample = sample[newaxis, :]
    melfbanks = MelFilterbanks(sample_rate=rate)
    attr = melfbanks(sample)
    attr = np.array(attr).T

    mfcc_audios.append(attr)

mfcc_audios = np.array(mfcc_audios)

mfcc_audios = scaler.transform(
    mfcc_audios.reshape(-1, mfcc_audios.shape[-1])).reshape(mfcc_audios.shape)

prediction = model.predict(mfcc_audios)

# true_pred = prediction > 0.25

# y_hats = []
# confidences = []
# mapping = []

# for j, arr in enumerate(true_pred):
#     y_hats.append([])
#     confidences.append([])
#     mapping.append([])
#     for i, isTrue in enumerate(arr):
#         if isTrue:
#             y_hats[j].append(i)
#             confidences[j].append(prediction[j][i])
#             mapping[j].append(info['mapping'][i])

# print(y_hats)
# print(confidences)
# print(mapping)

# for i, arr in enumerate(y_hats):
#     print(f'Confiança: {confidences[i]} | Predito: { mapping[i]}')

y_hat = np.argmax(prediction, axis=-1)
confidence = np.max(prediction, axis=-1)

for i, arr in enumerate(y_hat):
    print(f'Confiança: {confidence[i]} | Predito: { info["mapping"][arr]}')
