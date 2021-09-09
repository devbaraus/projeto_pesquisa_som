#!/usr/bin/env python
# coding: utf-8
# %%
from numpy import random
import tensorflow as tf
import sys
from python_speech_features import mfcc
from scipy.signal.windows import hann
from librosa import stft
import multiprocessing
import numpy as np
from joblib import Parallel, delayed


from deep_audio import Directory, Audio, Process, Terminal, Augmentation, Visualization
args = Terminal.get_args(sys.argv[1:])

if 'melbanks' == args['representation']:
    from leaf_audio.frontend import MelFilterbanks
    from tensorflow import newaxis

# %%


# %%
num_cores = multiprocessing.cpu_count()
# amostra do sinal
sampling_rate = 24000
# quantidade de segmentos
n_segments = args['segments'] or None
# quantidade de audios
n_audios = args['people'] or None
# bibliotecas
library = args['representation']
# lingua
language = args['language'] or 'portuguese'
# normalização do sinal
normalization = args['normalization'] or 'nonorm'
# flat processing
flat = args['flat']
# caminho para os audios
path = f'{language}/audios/{sampling_rate}'
# augmentation
augment = args['augmentation']

f = Directory.filenames(path)

# %%


def _noise(sample, rate):
    mean_intensity = np.mean(np.abs(sample))
    intensity = np.mean(np.abs(sample)[np.abs(sample) > mean_intensity])

    return Augmentation.noise_addition(
        sample, random.uniform(intensity * 0.3, intensity * 0.8))


def _cut(sample, rate):
    cut_seconds = random.randint(sample.shape[0] * 0.2, sample.shape[0] * 0.6)
    pos_cut = random.randint(
        sample.shape[0] * 0.1, sample.shape[0] * 0.9)

    return Augmentation.cut_signal(sample, pos_cut, cut_seconds)


def process_directory(dir, index, library):
    signal, rate = Audio.read(
        f'{path}/{dir}', sr=sampling_rate, normalize=True)

    signal = np.array(signal)

    segment_time = 5

    # arredonda o sinal de audio para multiplo de 5
    signal = signal[:len(signal) - len(signal) % (rate * segment_time)]

    # avalia quantos segmentos têm em uma audio
    segments = len(signal) // (rate * segment_time)

    augsize = int(augment[0]) if len(augment) > 0 else 0

    m = {
        'attrs': [],
        'labels': [index] * (n_segments or segments) * (1 + augsize),
        'classes': [{index: (n_segments or segments) * (1 + augsize)}]
    }

    for i in range(segments):
        if n_segments and i >= n_segments:
            continue

        samples = [Audio.segment(signal, rate, seconds=segment_time, window=i)]

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

        # if augment:
        #     for _ in range(int(augment[0])):
        #         flag = False
        #         aug = samples[0]

        #         if random.uniform() > 0.5 and 'cut' in augment:
        #             aug = _cut(aug, rate)
        #             flag = True

        #         if random.uniform() > 0.5 and 'noise' in augment:
        #             aug = _noise(aug, rate)
        #             flag = True

        #         if not flag and len(augment) == 3:
        #             if random.uniform() > 0.5:
        #                 aug = _cut(aug, rate)
        #             else:
        #                 aug = _noise(aug, rate)

        #         samples.append(aug)

        for sample_index, sample in enumerate(samples):
            if library == 'stft':
                attr = np.abs(
                    np.array(stft(sample, n_fft=n_fft, hop_length=hop_length)))

            if library == 'melbanks':
                sample = sample[newaxis, :]
                melfbanks = MelFilterbanks(sample_rate=rate)
                attr = melfbanks(sample)
                attr = np.array(attr).T

            if library == 'psf':
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
                attr = np.array(attr)

            Visualization.plot_cepstrals(
                attr, fig_name=f'teste.png')
            Audio.write(
                f'portuguese/processed/psf/{dir}_{i}_{sample_index}.wav', sample, rate)

            m['attrs'].append(attr.tolist())

        del attr
    del signal
    return m


if __name__ == '__main__':
    filename = Directory.processed_filename(
        language, library, sampling_rate, n_audios, n_segments, augment)

    # m = []
    # for j, i in enumerate(f):
    #     if j < 1:
    #         m.append(process_directory(i, j, library))

    m = Parallel(n_jobs=-1, verbose=len(f))(
        delayed(process_directory)
        (i, j, library)
        for j, i in enumerate(f)
        if n_audios == None or j < n_audios
    )

    Process.object_to_json(
        filename,
        m,
        f,
    )
    del m
