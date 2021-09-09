from numpy.core.fromnumeric import squeeze


def object_to_json(filename, attrs, files):
    from deep_audio import JSON

    data = {
        'mapping': [file.replace('.wav', '') for _, file in enumerate(files)],
        'classes': [],
        'labels': [],
        'attrs': []
    }

    for i in attrs:
        data['attrs'].extend(i['attrs'])
        data['labels'].extend(i['labels'])
        data['classes'].extend(i['classes'])

    JSON.create_json_file(filename, data, cls=JSON.NumpyEncoder)

    del data


def object_to_attention(filename, attrs, files):
    from deep_audio import Directory
    data = {
        'labels': [],
        'attrs': [],
        'mapping': [file.replace('.wav', '') for _, file in enumerate(files)]
    }

    for i in attrs:
        data['attrs'].extend(i['attrs'])
        data['labels'].extend(i['labels'])

    rows = []

    for info, i in enumerate(data['labels']):
        row = f'{info} qid:{info} '
        info_attrs = flatten_matrix(data['attrs'][i])
        for info_attr, j in enumerate(info_attrs):
            row += f'{j}:{info_attr} '
        rows.append(row)

    Directory.create_file(filename, rows)
    del data


def pad_accuracy(acc, pad=4):
    return str(int(acc * 10000)).zfill(pad)


def flatten_matrix(signal_matrix):
    from numpy import array

    matrix_holder = []
    for row in signal_matrix:
        matrix_holder.append(row.flatten())

    return array(matrix_holder)


def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def selection(folder, valid_size=0.25, test_size=0.2, random_state=42, flat=False, squeeze=False, mapping=False):
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import squeeze

    X, y, labels = Directory.load_json_data(folder)

    if flat:
        X = flatten_matrix(X)

    if squeeze == True:
        X = squeeze(X, axis=3)

    if test_size == 0:
        if mapping:
            return X, y, labels

        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    if valid_size == 0:
        return X_train, X_test, y_train, y_test

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          stratify=y_train,
                                                          test_size=valid_size,
                                                          random_state=random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def mixed_selection(first_folder, second_folder, third_folder, fourth_folder, lm_validation=False, lm_test=False, rm_validation=False, rm_test=False,
                    valid_size=0.25,
                    test_size=0.2,
                    random_state=42):
    global X_train, y_train, X_valid, y_valid, X_test, y_test
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import concatenate
    import numpy as np

    X_first, y_first, _ = Directory.load_json_data(first_folder)
    X_second, y_second, _ = Directory.load_json_data(second_folder)
    X_third, y_third, _ = Directory.load_json_data(third_folder)
    X_fourth, y_fourth, _ = Directory.load_json_data(fourth_folder)

    X_first = flatten_matrix(X_first)
    X_second = flatten_matrix(X_second)
    X_third = flatten_matrix(X_third)
    X_fourth = flatten_matrix(X_fourth)

    X_train_first, X_test_first, y_train_first, y_test_first = train_test_split(X_first,
                                                                                y_first,
                                                                                stratify=y_first,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    X_train_first, X_valid_first, y_train_first, y_valid_first = train_test_split(X_train_first,
                                                                                  y_train_first,
                                                                                  stratify=y_train_first,
                                                                                  test_size=valid_size,
                                                                                  random_state=random_state)

    X_train_second, X_test_second, y_train_second, y_test_second = train_test_split(X_second,
                                                                                    y_second,
                                                                                    stratify=y_second,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

    X_train_second, X_valid_second, y_train_second, y_valid_second = train_test_split(X_train_second,
                                                                                      y_train_second,
                                                                                      stratify=y_train_second,
                                                                                      test_size=valid_size,
                                                                                      random_state=random_state)

    X_train_first = concatenate((X_train_first, X_train_second), axis=1)
    y_train = y_train_first

    if not rm_validation:
        X_valid_first = X_valid_first
    else:
        X_valid_first = concatenate((X_valid_first, X_valid_second), axis=1)

    y_valid = y_valid_first

    if not rm_test:
        X_test_first = X_test_first
    else:
        X_test_first = concatenate((X_test_first, X_test_second), axis=1)

    y_test = y_test_first

    X_train_third, X_test_third, y_train_third, y_test_third = train_test_split(X_third,
                                                                                y_third,
                                                                                stratify=y_third,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    X_train_third, X_valid_third, y_train_third, y_valid_third = train_test_split(X_train_third,
                                                                                  y_train_third,
                                                                                  stratify=y_train_third,
                                                                                  test_size=valid_size,
                                                                                  random_state=random_state)

    X_train_fourth, X_test_fourth, y_train_fourth, y_test_fourth = train_test_split(X_fourth,
                                                                                    y_fourth,
                                                                                    stratify=y_fourth,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

    X_train_fourth, X_valid_fourth, y_train_fourth, y_valid_fourth = train_test_split(X_train_fourth,
                                                                                      y_train_fourth,
                                                                                      stratify=y_train_fourth,
                                                                                      test_size=valid_size,
                                                                                      random_state=random_state)

    X_train_third = concatenate((X_train_third, X_train_fourth), axis=1)

    if not rm_validation:
        X_valid_third = X_valid_third
    else:
        X_valid_third = concatenate((X_valid_third, X_valid_fourth), axis=1)

    if not rm_test:
        X_test_third = X_test_third
    else:
        X_test_third = concatenate((X_test_third, X_test_fourth), axis=1)

    X_train = concatenate((X_train_first, X_train_third), axis=0)
    y_train = concatenate(
        (y_train_first, y_train_third + np.max(y_train_first) + 1), axis=0)

    if not lm_validation:
        X_valid = X_valid_first
        y_valid = y_valid_first
    else:
        X_valid = concatenate((X_valid_first, X_valid_third), axis=0)
        y_valid = concatenate(
            (y_valid_first, y_valid_third + np.max(y_valid_first) + 1), axis=0)

    if not lm_test:
        X_test = X_test_first
        y_test = y_test_first
    else:
        X_test = concatenate((X_test_first, X_test_third), axis=0)
        y_test = concatenate(
            (y_test_first, y_test_third + np.max(y_test_first) + 1), axis=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def mixed_selection_representation(first_folder, second_folder, validation=False, test=False,
                                   valid_size=0.25,
                                   test_size=0.2,
                                   random_state=42):
    global X_train, y_train, X_valid, y_valid, X_test, y_test
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import concatenate

    X_portuguese, y_portuguese, _ = Directory.load_json_data(first_folder)
    X_english, y_english, _ = Directory.load_json_data(second_folder)

    X_portuguese = flatten_matrix(X_portuguese)
    X_english = flatten_matrix(X_english)

    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(X_portuguese,
                                                                    y_portuguese,
                                                                    stratify=y_portuguese,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_pt, X_valid_pt, y_train_pt, y_valid_pt = train_test_split(X_train_pt,
                                                                      y_train_pt,
                                                                      stratify=y_train_pt,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_english,
                                                                    y_english,
                                                                    stratify=y_english,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_en, X_valid_en, y_train_en, y_valid_en = train_test_split(X_train_en,
                                                                      y_train_en,
                                                                      stratify=y_train_en,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train = concatenate((X_train_pt, X_train_en), axis=1)
    y_train = y_train_pt

    if not validation:
        X_valid = X_valid_pt
    else:
        X_valid = concatenate((X_valid_pt, X_valid_en), axis=1)

    y_valid = y_valid_pt

    if not test:
        X_test = X_test_pt
    else:
        X_test = concatenate((X_test_pt, X_test_en), axis=1)

    y_test = y_test_pt

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def mixed_selection_language(portuguese_folder, english_folder, validation=False, test=False, valid_size=0.25,
                             test_size=0.2,
                             random_state=42, flat=False, squeeze=False):
    global X_train, y_train, X_valid, y_valid, X_test, y_test
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import concatenate, squeeze, max

    X_portuguese, y_portuguese, _ = Directory.load_json_data(
        portuguese_folder)
    X_english, y_english, _ = Directory.load_json_data(english_folder)

    if flat:
        X_portuguese = flatten_matrix(X_portuguese)
        X_english = flatten_matrix(X_english)

    # if squeeze:
    #     X_portuguese = squeeze(X_portuguese, axis=3)
    #     X_english = squeeze(X_english, axis=3)

    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(X_portuguese,
                                                                    y_portuguese,
                                                                    stratify=y_portuguese,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_pt, X_valid_pt, y_train_pt, y_valid_pt = train_test_split(X_train_pt,
                                                                      y_train_pt,
                                                                      stratify=y_train_pt,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_english,
                                                                    y_english,
                                                                    stratify=y_english,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_en, X_valid_en, y_train_en, y_valid_en = train_test_split(X_train_en,
                                                                      y_train_en,
                                                                      stratify=y_train_en,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train = concatenate((X_train_pt, X_train_en), axis=0)
    y_train = concatenate(
        (y_train_pt, y_train_en + max(y_train_pt) + 1), axis=0)

    if not validation:
        X_valid = X_valid_pt
        y_valid = y_valid_pt
    else:
        X_valid = concatenate((X_valid_pt, X_valid_en), axis=0)
        y_valid = concatenate(
            (y_valid_pt, y_valid_en + max(y_valid_pt) + 1), axis=0)

    if not test:
        X_test = X_test_pt
        y_test = y_test_pt
    else:
        X_test = concatenate((X_test_pt, X_test_en), axis=0)
        y_test = concatenate(
            (y_test_pt, y_test_en + max(y_test_pt) + 1), axis=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
