#!/bin/bash

for i in `seq 0 2`; do
  echo Launching container N. $i
  docker run -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test-label > "label-${i}.out" &
done

wait
echo Completed
