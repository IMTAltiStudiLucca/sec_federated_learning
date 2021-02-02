#!/bin/bash

if [[ $1 -gt $(nproc) ]]
then
  echo "Not enough CPUs"
  exit
fi

index=$2

for i in `seq 0 $(($1 - 1))`; do
  setup="./simulations/simulazione_${$((index + i))}.yaml"
  mkdir -p "./${i}"
  echo Launching container N. $i with setup $setup
  docker run --cpuset-cpus $((i)) --env script=label-attack.py --env setup=$setup -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test >> "./${i}/simulazione_${$((index + i))}.out" 2>&1 &
done
wait
echo Completed
