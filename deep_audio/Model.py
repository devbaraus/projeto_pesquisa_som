import numpy as np


def create_model_json_file(file, data):
    from deep_audio import Directory

    directory = '/'.join(file.split('/')[:-1])

    directory.create_directory(directory)

    with open(file, "w") as json_file:
        json_file.write(data)


def dump_grid(file, model, language, method, seed, library, sizes, score_train, score_test, sampling_rate, score_valid=None,
              model_file=None, extra={}):
    from time import time
    from deep_audio import JSON

    dump_info = {
        'method': method,
        'language': language,
        'seed': seed,
        'library': library,
        'sample_rate': sampling_rate,
        'sizes': sizes,
        'score_train': score_train,
        'score_test': score_test,
        'timestamp': time(),
        'params': model.best_params_,
        'cv_results': model.cv_results_,
        **extra
    }

    if score_valid:
        dump_info['score_valid'] = score_valid

    if model:
        dump_info['model_file'] = model_file

    JSON.create_json_file(file, dump_info, cls=JSON.NumpyEncoder)

    return


def load_processed_data(path, inputs_fieldname='mfcc'):
    import json
    import numpy as np

    with open(path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data[inputs_fieldname])
    targets = np.array(data['labels'])
    mapping = data['mapping']

    return inputs, targets, mapping
