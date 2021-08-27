import numpy as np
import librosa


def noise_addition(signal, intensity):
    # Noise addition using normal distribution with mean = 0 and std =1
    # Permissible noise factor value = x > 0.004

    return signal + intensity * np.random.uniform(low=-1.0, high=1.0, size=len(signal))


def cut_signal(signal, start_index, size_seconds):
    # Add silence to a position of the signal
    sample = signal.copy()

    if start_index >= len(sample):
        raise ValueError(
            "start_silence is equal or bigger than the signal length")

    if size_seconds >= len(sample):
        raise ValueError(
            "silence_seconds is equal or bigger than the signal length")

    total_size = min(size_seconds, sample.shape[0] - start_index)
    zeros_seconds = np.zeros(total_size)
    sample[start_index: start_index + total_size] = zeros_seconds
    return sample


def time_shifting(signal, rate):
    # Shifting the sound wave
    # Permissible factor values = sr/10

    return np.roll(signal, rate // 10)


def time_stretching(signal, factor=0.4):
    # Permissible factor values = 0 < x < 1.0
    # If factor is 0.4, then the signal is stretched by a factor of 0.4x

    return librosa.effects.time_stretch(signal, factor)


def pitch_shifting(signal, rate, n_steps=-5):
    # Permissible factor values = -5 <= x <= 5
    # If n_steps is -5, then the signal is shifted by 5 semitones

    return librosa.effects.pitch_shift(signal, rate, n_steps=n_steps)
