#!/bin/bash
INPUT=/kaggle/input/jigsaw-multilingula-toxicity-token-encoded
if [ ! -d $INPUT/XNLI ]; then
    curl -O https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
    curl -O https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip
    curl -O https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

    extrace_to_folder() {
        zip_name=$1
        zn=${zip_name//.zip/}
        #mkdir "$zn" && pushd "$zn" && unzip $OLDPWD/"$zip_name" && popd
        unzip -d "$zn" "$zip_name"
    }
    export -f extrace_to_folder

    find . -name "*.zip" -print0 | xargs -0 -I{} bash -xc 'extrace_to_folder {}'

    DEV_TEST=XNLI-1.0
    MT_TRAIN=XNLI-MT-1.0

    mkdir XNLI && mv $DEV_TEST/* XNLI && mv $MT_TRAIN/* XNLI
    rmdir "$DEV_TEST"
    rmdir "$MT_TRAIN"

    export BERT_BASE_DIR=$PWD/multi_cased_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
    export XNLI_DIR=$PWD/XNLI/XNLI-MT-1.0/
else
    export BERT_BASE_DIR=$INPUT/multi_cased_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
    export XNLI_DIR=$INPUT/XNLI/XNLI-MT-1.0/
fi

#test our datasets

STAGE="extract_feature"

if [ $STAGE = "extract_feature" ]; then

    TI=/tmp/input.txt
    if [ ! -f $TI ]; then
        python kaggle_runner/datasets/jigsaw_toxic_data.py
        sed -i 's/"\{1,3\}\s*\(.*\)\s*"\{1,3\}$/\1/' $TI
        wc_l_info=$(wc -l $TI)
    fi

    echo "head of the comments list:"
    head /tmp/input.txt
    echo "lines info: $wc_l_info"

    git clone https://github.com/pennz/bert
    cd bert || exit 1

    export STORAGE_BUCKET=gs://kaggle_runner/
    export TASK_NAME=toxic

    if [ ! -z $TPU_NAME ]; then
        #export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12
        #export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/multi_cased_L-12_H-768_A-12
        TPU_Parameter="--use_tpu=True --tpu_name=$TPU_NAME --output_dir=${STORAGE_BUCKET}/${TASK_NAME}"
        OUT_PARA="--output_file=/home/multi_cased_features.jsonl"
    else
        #export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/multi_cased_L-12_H-768_A-12
        OUT_PARA="--output_file=/home/multi_cased_features.jsonl"
    fi

    python extract_features.py \
        --input_file=/tmp/input.txt \
        --vocab_file="$BERT_BASE_DIR/vocab.txt" \
        --bert_config_file="$BERT_BASE_DIR/bert_config.json" \
        --init_checkpoint="$BERT_BASE_DIR/bert_model.ckpt" \
        --layers=-1,-2,-3,-4 \
        --max_seq_length=128 $OUT_PARA $TPU_Parameter \
        --batch_size=32

else
    python run_classifier.py \
        --task_name=XNLI \
        --do_train=true \
        --do_eval=true \
        --data_dir="$XNLI_DIR" \
        --vocab_file="$BERT_BASE_DIR/vocab.txt" \
        --bert_config_file="$BERT_BASE_DIR/bert_config.json" \
        --init_checkpoint="$BERT_BASE_DIR/bert_model.ckpt" \
        --do_lower_case=False \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=5e-5 \
        --num_train_epochs=0.5 \
        --output_dir=/tmp/xnli_output/
fi
