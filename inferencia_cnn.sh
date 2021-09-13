#!/bin/bash
echo "arquivo,  memoria (KB), tempo (m:s), confiança, predito" >> $2

for i in $1/*.wav; do
  /usr/bin/time -v python3 inferencia_cnn.py -l portuguese -r melbanks -n standard -a 15,noise,cut -i "$i" 2>&1 | tee temp_inf.txt
  wait < /dev/tty
  mem=$(cat temp_inf.txt | grep "Maximum resident set size (kbytes):" | cut -d " " -f 6)
  elapsed=$(cat temp_inf.txt | grep "Elapsed (wall clock) time (h:mm:ss or m:ss):" | cut -d " " -f 8)
  name=$(echo $i | cut -d "/" -f 3)
  predito=$(cat temp_inf.txt | grep "Predito:" | cut -d " " -f 2,5 --output-delimiter=',')

  for j in $predito; do
    echo "$name, $mem, $elapsed, $j" >> $2
  done
done