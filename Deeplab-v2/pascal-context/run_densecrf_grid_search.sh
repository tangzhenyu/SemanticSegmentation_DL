#!/bin/bash 

DATASET=voc12
LOAD_MAT_FILE=1
MODEL_NAME=deelab_largeFOV

TEST_SET=val
FEATURE_TYPE=fc8

# how many images used for cross-validation
NUM_SAMPLE=100

# default values
MAX_ITER=10

POS_X_STD=3
POS_Y_STD=3
POS_W=3

Bi_X_STD=50
Bi_Y_STD=50
Bi_R_STD=10
Bi_G_STD=10
Bi_B_STD=10
Bi_W=5

# We did not cross-validate for pos_w and pos_xy_std

range_POS_W=(3)
range_POS_XY_STD=(3)

# SPECIFY the GRID SEARCH RANGE
range_W=(4)
range_XY_STD=(51)
range_RGB_STD=(5)

for posW in ${range_POS_W[@]}
do
    POS_W=${posW}
    
    for posXY in ${range_POS_XY_STD[@]}
    do
	POS_X_STD=${posXY}
	POS_Y_STD=${posXY}

	for valW in ${range_W[@]}
	do
	    Bi_W=${valW}

	    for valX in ${range_XY_STD[@]}
	    do
		Bi_X_STD=${valX}
		Bi_Y_STD=${valX}

		for valR in ${range_RGB_STD[@]}
		do
		    Bi_R_STD=${valR}
		    Bi_G_STD=${valR}
		    Bi_B_STD=${valR}
	    
#######################################
# MODIFY THE PATH FOR YOUR SETTING
#######################################
		    CRF_DIR=/rmt/work/deeplab/code/densecrf

		    if [ ${LOAD_MAT_FILE} == 1 ]
		    then
			CRF_BIN=${CRF_DIR}/prog_refine_pascal_v4
			FILE_NAME=mat
		    else
			CRF_BIN=${CRF_DIR}/prog_refine_pascal
			FILE_NAME=bin
		    fi

		    if [ ${DATASET} == "voc12" ]
		    then
			IMG_DIR_NAME=pascal/VOCdevkit/VOC2012
		    elif [ ${DATASET} == "coco" ]
		    then
			IMG_DIR_NAME=coco
                    elif [ ${DATASET} == "voc10_part" ]
                    then
                        IMG_DIR_NAME=pascal/VOCdevkit/VOC2012
		    fi
	    
		    IMG_DIR=/rmt/data/${IMG_DIR_NAME}/PPMImages

		    # FEATURE_DIR saves the features for those "num_sample" images		    
		    FEATURE_DIR=/rmt/work/deeplab/exper/${DATASET}/features/${MODEL_NAME}/${TEST_SET}/${FEATURE_TYPE}/${FILE_NAME}_numSample${NUM_SAMPLE}
		    
		    SAVE_DIR=/rmt/work/deeplab/exper/${DATASET}/res/features/${MODEL_NAME}/${TEST_SET}/${FEATURE_TYPE}/post_densecrf_W${Bi_W}_XStd${Bi_X_STD}_RStd${Bi_R_STD}_PosW${POS_W}_PosXStd${POS_X_STD}_numSample${NUM_SAMPLE}

		    mkdir -p ${SAVE_DIR}

		    # run the program
		    ${CRF_BIN} -id ${IMG_DIR} -fd ${FEATURE_DIR} -sd ${SAVE_DIR} -i ${MAX_ITER} -px ${POS_X_STD} -py ${POS_Y_STD} -pw ${POS_W} -bx ${Bi_X_STD} -by ${Bi_Y_STD} -br ${Bi_R_STD} -bg ${Bi_G_STD} -bb ${Bi_B_STD} -bw ${Bi_W}
		done
	    done
	done
    done
done
