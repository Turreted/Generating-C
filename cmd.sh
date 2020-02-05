#!/bin/bash

for file in $(find . -name '*.c' -o -name '*.h')
do
	cat $file >> /Users/colin.mitchell/desktop/python/keras/C_gen/massive_c.txt && echo $file
done
