def resample(data, current_rate, new_rate):
    from librosa import resample

    data = resample(data, current_rate, new_rate)
    return data, new_rate


def read(path, sr=None, mono=True, normalize=True):
    from librosa import load

    data, rate = load(path, sr=sr, mono=mono)

    if not normalize:
        data = unnormalize(data)

    return data, rate


def write(path, data, rate):
    import soundfile as sf
    from deep_audio import Directory

    Directory.create_directory(path, file=True)

    sf.write(path, data, rate, subtype='PCM_16')


def to_mono(data):
    from numpy import mean
    if len(data.shape) > 1 and data.shape[1] > 0:
        data = mean(data, axis=1, dtype=type(data[0][0]))
    return data


def db(data, n_fft=2048):
    from librosa import stft, amplitude_to_db
    from numpy import abs, max
    S = stft(data, n_fft=n_fft, hop_length=n_fft // 2)
    D = amplitude_to_db(abs(S) * 1, max)
    return max(abs(D))


def normalize(data):
    from numpy import float32, int16
    data_type = type(data[0])

    if data_type == int16:
        data = data.astype(float32) / 32768

    return data


def unnormalize(data):
    from numpy import float32, array, int16
    data_type = type(data[0])

    if data_type == float32:
        data = array(data * 32768).astype(int16)

    return data


def segment(signal, rate, seconds, window=0):
    start_sample = rate * window * seconds
    finish_sample = start_sample + (rate * seconds)

    return signal[start_sample:finish_sample]


def trim(signal, top_db=20):
    from librosa.effects import split

    intervals = split(signal, top_db=20)

    return signal[intervals[0][0]:intervals[-1][-1]]
