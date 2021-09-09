# %%
from pickle import dump
from deep_audio import Directory, Process, Terminal, Model
from sklearn.svm import SVC

# %%
args = Terminal.get_args()

language = args['language']
method = 'svm'
library = args['representation']
people = args['people']
segments = args['segments']
normalization = args['normalization']
augment = args['augmentation']
sampling_rate = 24000
random_state = 42

# %%
file_path = Directory.processed_filename(
    language, library, sampling_rate, people, segments, augment)
# %%
X_train, y_train, mapping = Process.selection(
    file_path, valid_size=0, test_size=0, mapping=True, flat=True)

# %%
if normalization == 'minmax':
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

elif normalization == 'standard':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

model = SVC(C=10, kernel='linear', decision_function_shape='ovo')

filename_holder = Directory.model_filename(
    method=method,
    language=language,
    library=library,
    normalization=normalization,
    augmentation=augment,
    json=False,
    models=True
)

model.fit(X_train, y_train)

Directory.create_directory(filename_holder)
dump(model, open(filename_holder + 'weight.h5', 'wb'))
dump(scaler, open(filename_holder + 'scaler.pkl', 'wb'))

# SALVA OS PARAMETROS
Model.dump_model(
    filename_holder + 'info.json',
    language=language,
    method=method,
    normalization=normalization,
    sampling_rate=sampling_rate,
    augmentation=augment,
    shape=X_train.shape,
    seed=random_state,
    library=library,
    extra={
        'mapping': mapping
    }
)
