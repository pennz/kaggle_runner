#!/bin/bash
INPUT=/kaggle/input/jigsaw-multilingula-toxicity-token-encoded
if [ ! -d $INPUT/XNLI ]; then
    curl -O https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip
    curl -O https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip
    curl -O https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

    extrace_to_folder () {
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
    export XNLI_DIR=$PWD/XNLI
else
    export BERT_BASE_DIR=$INPUT/multi_cased_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
    export XNLI_DIR=$INPUT/XNLI
fi

#test our datasets
git clone --depth=1 https://github.com/ultrons/bert
#https://github.com/google-research/bert
#tf_upgrade_v2 --intree bert/ --outtree bert_v2 --reportfile report.txt
cd bert

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
--num_train_epochs=0.1 \
--output_dir=/tmp/xnli_output/
