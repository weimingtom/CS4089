#!/bin/bash -e

[[ -z "$1" || -z "$2" ]] && (echo "Usage: $0 <input> <output>" && exit 1)

mplayer $1 -ao pcm:file=$2 -loop 1
