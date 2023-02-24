#!/bin/bash
i=0
j=1
for x in {0..7}
do
	echo $i,$j;
	python3 ./3_test_models.py -idx1 $i -idx2 $j
	i=$((i+2));
	j=$((j+2));

done
