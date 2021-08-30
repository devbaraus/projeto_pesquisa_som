# %%
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from deep_audio import Directory, Process, Terminal, Model
# %%
args = Terminal.get_args()

language = args['language']
library = args['representation']
people = args['people']
segments = args['segments']
normalization = args['normalization']
augment = args['augmentation']
sampling_rate = 24000
random_state = 42
# %%
global X_train, X_valid, X_test, y_train, y_valid, y_test

file_path = Directory.processed_filename(
    language, library, sampling_rate, people, segments, augment)
# %%

X_train, X_valid, X_test, y_train, y_valid, y_test = Process.selection(
    file_path, flat=True)

if normalization == 'minmax':
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

elif normalization == 'standard':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

param_grid = {
    'C': [10],
    'kernel': ['linear'],
    'decision_function_shape': ['ovo']
}

model = GridSearchCV(svm.SVC(), param_grid, cv=5,
                     refit=True, verbose=2, n_jobs=-1)

model.fit(X_train, y_train)

best_params = model.best_params_

score_test = model.score(X_test, y_test)

score_train = model.score(X_train, y_train)

y_hat = model.predict(X_test)

filename_ps = Directory.verify_people_segments(
    people=people, segments=segments)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_grid(
    Directory.model_filename(
        'svm', language, library, normalization, score_test, augmentation=augment),
    model=model,
    language=language,
    method='Support Vector Machines',
    normalization=normalization,
    sampling_rate=sampling_rate,
    augmentation=augment,
    shape=X_train.shape,
    seed=random_state,
    library=library,
    sizes=[len(X_train), len(X_valid), len(X_test)],
    score_train=score_train,
    score_test=score_test,
)
