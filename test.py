from matplotlib.pyplot import show
from python_speech_features import mfcc
from scipy.signal.windows import hann
from deep_audio import Audio, Visualization

signal, rate = Audio.read(
    'inferencia/hugo/Frase 1-1.wav', 24000)

rate = 24000

n_mfcc = 40
n_mels = 40
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
    signal=signal,
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

Visualization.plot_cepstrals(
    attr, fig_name="./normal_40.png")
