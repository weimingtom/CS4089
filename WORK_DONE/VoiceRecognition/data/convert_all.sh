#!/bin/bash -e

mkdir -p $1/converted
for i in $1/*.wav; do
	echo $i
	$(dirname $BASH_SOURCE)/wav_format.sh $i $1/converted/$i 2>/dev/null 1>/dev/null
done
