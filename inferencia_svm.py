from deep_audio.Process import flatten_matrix
import json
import numpy as np
from tensorflow import keras
from deep_audio import Terminal, Directory, Audio
from python_speech_features import mfcc
from scipy.signal.windows import hann
from pickle import load
import librosa

args = Terminal.get_args()

language = 'portuguese'
method = 'svm'
library = 'psf'
people = args['people']
segments = args['segments']
normalization = args['normalization']
augment = args['augmentation']
sampling_rate = 24000
random_state = 42

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

model = load(open(filename_holder + 'model.h5', 'rb'))

signal, rate = librosa.load(args['inferencia'], sr=sampling_rate)

# signal = Audio.trim(signal)

segment_time = 5
signal = signal[:len(signal) - len(signal) % (rate * segment_time)]

segments = len(signal) // (rate * segment_time)

mfcc_audios = []

for i in range(segments):
    sample = Audio.segment(signal, rate, seconds=segment_time, window=i)

    n_mfcc = 13
    n_mels = 26
    n_fft = 2048
    # Janela e overlapping (em amostras)
    hop_length = 512
    win_length = 1024
    # Janela e overlapping (em tempo)
    win_len = win_length / rate
    win_hop = hop_length / rate
    lifter = 22
    fmin = 0
    fmax = rate / 2
    coef_pre_enfase = 0.97
    append_energy = 0

    attr = mfcc(
        signal=sample,
        samplerate=rate,
        winlen=win_len,
        winstep=win_hop,
        numcep=n_mfcc,
        nfilt=n_mels,
        nfft=n_fft,
        lowfreq=fmin,
        highfreq=fmax,
        preemph=coef_pre_enfase,
        ceplifter=lifter,
        appendEnergy=append_energy,
        winfunc=hann
    )

    mfcc_audios.append(attr)

mfcc_audios = np.array(mfcc_audios)

mfcc_audios = flatten_matrix(mfcc_audios)

mfcc_audios = scaler.transform(
    mfcc_audios.reshape(-1, mfcc_audios.shape[-1])).reshape(mfcc_audios.shape)

prediction = model.predict(mfcc_audios)

for i, arr in enumerate(prediction):
    print(f'Predito: { info["mapping"][arr] }')
