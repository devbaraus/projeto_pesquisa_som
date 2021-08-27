# %%
from deep_audio import Audio, Visualization, Augmentation
import IPython.display as ipd
import librosa
import python_speech_features as psf
from scipy.signal.windows import hann
import numpy as np
# %%
signal, rate = Audio.read(
    'base_portuguese/2/p0984402b6241414d970ef97c4afba121_s00_a00.wav')

signal = Audio.segment(signal, rate, seconds=5, window=0)


n_mfcc = 20
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

# %%
Visualization.plot_audio(
    signal, rate, fig_name="test/normal_audio.png", show=True)
# Audio.write('test/normal_audio.wav', signal, rate)
# print(signal.shape)

# attr = psf.mfcc(
#     signal=signal,
#     samplerate=rate,
#     winlen=win_len,
#     winstep=win_hop,
#     numcep=n_mfcc,
#     nfilt=n_mels,
#     nfft=n_fft,
#     lowfreq=fmin,
#     highfreq=fmax,
#     preemph=coef_pre_enfase,
#     ceplifter=lifter,
#     appendEnergy=append_energy,
#     winfunc=hann
# )

# Visualization.plot_cepstrals(
#     attr, title="Normal", fig_name="test/nornal_mfcc.png", show=True)

# %%
noised_audio = Augmentation.noise_addition(signal, 0.02)
Visualization.plot_audio(
    noised_audio, rate, fig_name="test/cutted_audio.png", show=True)
# Audio.write('test/noised_audio.wav', noised_audio, rate)
# print(noised_audio.shape)

# attr = psf.mfcc(
#     signal=noised_audio,
#     samplerate=rate,
#     winlen=win_len,
#     winstep=win_hop,
#     numcep=n_mfcc,
#     nfilt=n_mels,
#     nfft=n_fft,
#     lowfreq=fmin,
#     highfreq=fmax,
#     preemph=coef_pre_enfase,
#     ceplifter=lifter,
#     appendEnergy=append_energy,
#     winfunc=hann
# )

# Visualization.plot_cepstrals(
#     attr, title="noised_audio", fig_name="test/noised_mfcc.png", show=True)

# %%
cutted_audio = Augmentation.cut_signal(
    signal, rate, 40779, 19565)
Visualization.plot_audio(
    cutted_audio, rate, fig_name="test/cutted_audio.png", show=True)

# %%
