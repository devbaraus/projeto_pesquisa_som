# python3 merge_audios.py -l portuguese
# python3 merge_audios.py -l english

# python3 process.py -l portuguese -r psf
# python3 process.py -l portuguese -r melbanks
# python3 process.py -l portuguese -r stft 


# python3 deepgrid.py -m cnn -l portuguese -r psf -n standard
# python3 deepgrid.py -m cnn -l portuguese -r psf -n minmax
# python3 deepgrid.py -m cnn -l portuguese -r psf -n nonorm
python3 deepgrid.py -m cnn -l portuguese -r melbanks -n standard
python3 deepgrid.py -m cnn -l portuguese -r melbanks -n minmax
python3 deepgrid.py -m cnn -l portuguese -r melbanks -n nonorm