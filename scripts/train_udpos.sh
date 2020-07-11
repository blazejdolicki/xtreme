# coding=utf-8
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='en'
# LANGS='kk,th,tl,yo'
NUM_EPOCHS=10
MAX_LENGTH=128
LR=2e-5

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
langs="de"
for lang in $langs; do
  TRAIN_LANG=$lang
  OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCH}-MaxLen${MAX_LENGTH}-train-${TRAIN_LANG}"
  mkdir -p $OUTPUT_DIR
  echo "Train language: ${lang}"
  python3 $REPO/third_party/run_tag.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --labels $DATA_DIR/labels.txt \
    --train_langs $TRAIN_LANG \
    --do_train \
    --do_eval \
    --do_predict_dev \
    --data_dir $DATA_DIR \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --max_seq_length  $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --save_steps 500 \
    --log_file $OUTPUT_DIR/train.log \
    --predict_langs $LANGS \
    --save_only_best_checkpoint $LC \
    --overwrite_output_dir \
    
    --seed 1 \
    --overwrite_cache \
    --logging_steps 500
done
