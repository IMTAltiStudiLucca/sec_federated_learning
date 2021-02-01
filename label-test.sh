#!/bin/bash

for i in `seq 1 6`; do
  echo Launching container N. $i
  docker run --cpus 1 -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test-label >> "label-${i}.out" 2>&1 &
done
wait
echo Completed
