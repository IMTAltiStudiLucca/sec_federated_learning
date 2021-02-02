#!/bin/bash

if [[ $1 -gt $(nproc) ]]
then
  echo "Not enough CPUs"
  exit
fi

for i in `seq 0 $(($1 - 1))`; do
  index=$(($2 + i))	
  setup="./simulations/simulazione_${index}.yaml"
  mkdir -p "./${i}"
  echo Launching container N. $i with setup $setup
  docker run --cpuset-cpus $((i)) --env script=score-attack.py --env setup=$setup -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test >> "./${index}/simulazione_${index}.out" 2>&1 &
done
wait
echo Completed
