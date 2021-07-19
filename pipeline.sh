# python3 merge_audios.py -l portuguese
# python3 merge_audios.py -l english

# python3 process.py -l portuguese -r psf
# python3 process.py -l portuguese -r melbanks
# python3 process.py -l portuguese -r stft
# python3 process.py -l portuguese -r psf,melbanks
# python3 process.py -l english -r psf,melbanks
# python3 process.py -l english -r psf,melbanks -s 65 -p 109
# python3 process.py -l english -r psf,melbanks -s 80 -p 80

# python3 gperceptron.py -l portuguese -r psf
python3 gperceptron.py -l portuguese -r stft
# python3 gperceptron.py -l portuguese -r melbanks
# python3 gperceptron.py -l portuguese -r mixed

# python3 gperceptron.py -l english -r psf
# python3 gperceptron.py -l english -r melbanks
# python3 gperceptron.py -l english -r mixed

# python3 gperceptron.py -l mixed -r psf 
# python3 gperceptron.py -l mixed -r melbanks 
# python3 gperceptron.py -l mixed -r mixed 

# python3 svm.py -l portuguese -r psf
# python3 svm.py -l portuguese -r stft
# python3 svm.py -l portuguese -r melbanks
# python3 svm.py -l portuguese -r mixed

# python3 svm.py -l english -r psf
# python3 svm.py -l english -r melbanks
# python3 svm.py -l english -r mixed

# python3 svm.py -l mixed -r psf
# python3 svm.py -l mixed -r melbanks
# python3 svm.py -l mixed -r mixed

# python3 cnn.py -l portuguese -r melbanks
python3 cnn.py -l portuguese -r stft