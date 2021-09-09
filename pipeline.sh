python3 merge_audios.py -l portuguese
# python3 merge_audios.py -l english

python3 process.py -l portuguese -r psf
python3 process.py -l portuguese -r psf -a 15,noise
# python3 process.py -l portuguese -r psf -a 5,noise,cut
# python3 process.py -l portuguese -r psf -a 10,noise,cut
# python3 process.py -l portuguese -r psf -a 15,noise,cut
# python3 process.py -l portuguese -r melbanks
# python3 process.py -l portuguese -r melbanks -a 5,noise,cut
# python3 process.py -l portuguese -r melbanks -a 10,noise,cut
# python3 process.py -l portuguese -r melbanks -a 15,noise,cut

# python3 process.py -l portuguese -r stft 


# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard
# python3 deepgrid.py -m cnn -l portuguese -r psf -n minmax
# python3 deepgrid.py -m cnn -l portuguese -r psf -n nonorm
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n minmax
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n nonorm

# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n minmax
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n nonorm

# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n nonorm
# python3 deepgrid.py -m lstm -l portuguese -r psf -n nonorm
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n minmax
# python3 deepgrid.py -m lstm -l portuguese -r psf -n minmax
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard

# python3 gperceptron.py -l english -r psf -n standard
# python3 gperceptron.py -l english -r psf -n minmax
# python3 gperceptron.py -l english -r melbanks -n standard
# python3 gperceptron.py -l english -r melbanks -n minmax

# python3 deepgrid.py -m cnn -l english -r psf -n standard
# python3 deepgrid.py -m cnn -l english -r psf -n minmax
# python3 deepgrid.py -m cnn -l english -r psf -n nonorm
# python3 deepgrid.py -m cnn -l english -r melbanks -n standard
# python3 deepgrid.py -m cnn -l english -r melbanks -n minmax
# python3 deepgrid.py -m cnn -l english -r melbanks -n nonorm

# python3 deepgrid.py -m lstm -l english -r melbanks -n nonorm
# python3 deepgrid.py -m lstm -l english -r psf -n nonorm
# python3 deepgrid.py -m lstm -l english -r melbanks -n minmax
# python3 deepgrid.py -m lstm -l english -r psf -n minmax
# python3 deepgrid.py -m lstm -l english -r melbanks -n standard
# python3 deepgrid.py -m lstm -l english -r psf -n standard


# python3 svm.py -l portuguese -r psf -n standard
# python3 svm.py -l portuguese -r psf -n standard -a 5,noise,cut
# python3 svm.py -l portuguese -r psf -n standard -a 10,noise,cut
# python3 svm.py -l portuguese -r psf -n standard -a 15,noise,cut

# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard
# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard -a 5,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard -a 10,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard -a 15,noise,cut

# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard
# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard -a 5,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard -a 10,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard -a 15,noise,cut

# python3 deepgrid.py -m perceptron -l portuguese -r psf -n standard
# python3 deepgrid.py -m perceptron -l portuguese -r psf -n standard -a 5,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r psf -n standard -a 10,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r psf -n standard -a 15,noise,cut


# python3 svm.py -l portuguese -r melbanks -n standard
# python3 svm.py -l portuguese -r melbanks -n standard -a 5,noise,cut
# python3 svm.py -l portuguese -r melbanks -n standard -a 10,noise,cut
# python3 svm.py -l portuguese -r melbanks -n standard -a 15,noise,cut

# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard -a 5,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard -a 10,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard -a 15,noise,cut

# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard -a 5,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard -a 10,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard -a 15,noise,cut

# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard -a 5,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard -a 10,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard -a 15,noise,cut


# python3 trainmodel.py -m cnn -l portuguese -r psf -n standard -a 15,noise,cut
# python3 trainmodel.py -m cnn -l portuguese -r melbanks -n standard -a 15,noise,cut
# python3 trainmodel.py -m perceptron -l portuguese -r psf -n standard -a 15,noise,cut
# python3 trainmodel.py -m perceptron -l portuguese -r melbanks -n standard -a 15,noise,cut
# python3 trainmodel.py -m svm -l portuguese -r psf -n standard -a 15,noise,cut
# python3 trainmodel.py -m svm -l portuguese -r melbanks -n standard -a 15,noise,cut

# python3 process.py -l portuguese -r psf -a 20,noise,cut
# python3 process.py -l portuguese -r melbanks -a 20,noise,cut

# python3 svm.py -l portuguese -r psf -n standard -a 20,noise,cut
# python3 svm.py -l portuguese -r melbanks -n standard -a 15,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard -a 20,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r psf -n standard -a 20,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r psf -n standard -a 20,noise,cut
# python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard -a 20,noise,cut
# python3 deepgrid.py -m lstm -l portuguese -r melbanks -n standard -a 20,noise,cut
# python3 deepgrid.py -m perceptron -l portuguese -r melbanks -n standard -a 20,noise,cut

python3 trainmodel_svm.py -l portuguese -r psf -n standard -a 15,noise