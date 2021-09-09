import json


def create_json_file(file, data, indent=2, cls=None):
    from deep_audio import Directory
    import json

    directory = '/'.join(file.split('/')[:-1])

    Directory.create_directory(directory)

    with open(file, "w") as fp:
        json.dump(data, fp, indent=indent, cls=cls)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        from numpy import ndarray

        if isinstance(obj, ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
