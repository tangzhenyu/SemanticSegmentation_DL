#!/usr/bin/env bash
# Martin Kersner, m.kersner@mail.com
# 2016/03/17

function help {
  if [ $# -ge 1 ]; then
    >&2 echo $1
  fi

  >&2 echo "Usage: ./jpg2ppm.sh INPUT_PATH OUTPUT_PATH"
  >&2 echo "INPUT_PATH denotes path containing JPG files for conversion."
  >&2 echo "OUTPUT_PATH denotes path where converted PPM files ar going to be saved."
}

if [ $# -eq 2 ]; then
  INPUT_PATH=$1
  OUTPUT_PATH=$2

  if [ -d $1 ] && [ -d $2 ]; then
    NUM_JPG_FILES=`ls $INPUT_PATH/*.jpg | wc -l`

    if [ $NUM_JPG_FILES -gt 0 ]; then
      # comment option clears any added comment to PPM image
      # if any comment was present DenseCRF crashed while reading such PPM image 
      mogrify -format ppm +comment -comment "" -path $OUTPUT_PATH $INPUT_PATH/*.jpg
    else
      help "Intput directory does not contain any JPG files!"
    fi

  else
    help "Input or output path does not exist!"
  fi
else
  help
fi
