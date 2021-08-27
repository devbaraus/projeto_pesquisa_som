import getopt
import sys


def get_args(argv=sys.argv[1:]):
    language = ''
    representation = ''
    method = ''
    normalization = 'nonorm'
    flat = False
    people = None
    segments = None
    augmentation = []
    try:
        opts, args = getopt.getopt(argv, "h:l:r:p:s:m:n:f:a:", [
                                   "language=", "representation=", "people=", "segments=", "method=", "normalization=", "flat=", "augmentation="])
    except getopt.GetoptError:
        print('test.py -l <language> -r <representation> -p <people> -s <segments>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -l <language> -r <representation> -p <people> -s <segments>')
            sys.exit()
        elif opt in ("-l", "--language"):
            language = arg
        elif opt in ("-r", "--representation"):
            representation = arg
        elif opt in ("-s", "--segments"):
            segments = int(arg)
        elif opt in ("-p", "--people"):
            people = int(arg)
        elif opt in ("-m", "--method"):
            method = arg
        elif opt in ("-n", "--normalization"):
            normalization = arg
        elif opt in ("-f", "--flat"):
            flat = bool(arg)
        elif opt in ("-a", "--augmentation"):
            augmentation = arg.split(',')
            if len(augmentation) < 2:
                raise Exception(
                    'Augmentation must be a list of two or more paths')

    args = {
        'language': language,
        'representation': representation,
        'people': people,
        'segments': segments,
        'method': method,
        'normalization': normalization,
        'flat': flat,
        'augmentation': augmentation,
    }

    print('ARGS:', args)

    return args
