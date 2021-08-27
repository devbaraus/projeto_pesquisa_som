# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:43:48 2021

@author: barau
"""

# %%
import librosa
import matplotlib.pyplot as plt
from SpecAugment import spec_augment_tensorflow
from deep_audio import Visualization, Audio
import tensorflow as tf
import numpy as np
import librosa.display
tf.compat.v1.enable_eager_execution()

# %%
audio_path = 'base_portuguese/1/p8f139305d25e4170a3d2cb5db79d9ed1_s00_a00.wav'

signal, sr = librosa.load(
    librosa.util.example_audio_file(), duration=5, sr=24000)
# %%
mel_spectrogram = librosa.feature.melspectrogram(
    y=signal, sr=sr, n_mels=256, hop_length=128, fmax=8000)

shape = mel_spectrogram.shape
mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))
# %%
mfcc = librosa.feature.melspectrogram(
    y=signal, sr=sr, n_mels=256, hop_length=128, fmax=8000)

shape = mfcc.shape
mfcc = np.reshape(mfcc, (-1, shape[0], shape[1], 1))

# %%
warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mfcc)
warped_masked_spectrogram = warped_masked_spectrogram.numpy()

# %%
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(
    warped_masked_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.tight_layout()
plt.title('SpecAugmented')
plt.show()
plt.close()
audio_signal = librosa.core.spectrum.griffinlim(
    warped_masked_spectrogram[0, :, :, 0])
Audio.write('test/warped_audio.wav', audio_signal, sr)
# %%
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(
    mfcc[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.tight_layout()
plt.title('MFCC')
plt.show()
plt.close()
audio_signal = librosa.core.spectrum.griffinlim(
    mfcc[0, :, :, 0])
Audio.write('test/mfcc_audio.wav', audio_signal, sr)
