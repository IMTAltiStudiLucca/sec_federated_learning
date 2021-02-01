#!/bin/bash

if [[ $1 -gt $(nproc) ]]
then
  echo "Not enough CPUs"
  exit
fi

for i in `seq 0 $(($1 - 1))`; do
  echo Launching container N. $i
  docker run --cpuset-cpus $((i)) -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test-label >> "label-${i}.out" 2>&1 &
done
wait
echo Completed
