# APIs
```sh
curl 'https://www.kaggle.com/requests/CancelKernelRunRequest' -H 'cookie: ka_sessionid=cc7e3a02c961799f3f7f281936848379; CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9kJehreeIrmSjmbQiumQNDP6K4pWAAyWOqWypIr_1-3cZm1KXwF4gMCGaMgbMQxLflFuS75NOWQC97EPRiGONyX4INqCDfF7cderxdVyFxUks5x9v6s4580M3CA-gdfYc; GCLB=CNeAmMPxnciELQ; _ga=GA1.2.1299897931.1589986631; _gid=GA1.2.519973049.1589986631; .ASPXAUTH=D4DD912AF0A259BC9170EBC507D60B2B1E932C9443B6D2FF4A604DB1E44DC38810688D6E075771E8D5BCF3CFA60AD0CF8EEC55FA8CABD17FE7B3F49777D80F5A32EE2209FAC1589A00EDBA88A46E17885F256939; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJrMWdhZ2dsZSIsIm5idCI6IjIwMjAtMDUtMjBUMTQ6NTg6NTUuODE5MTQxMloiLCJpYXQiOiIyMDIwLTA1LTIwVDE0OjU4OjU1LjgxOTE0MTJaIiwianRpIjoiNzgyODU5YzctM2YxMS00ZTllLThjOTEtZGNmYTg0NmI0MGNjIiwiZXhwIjoiMjAyMC0wNi0yMFQxNDo1ODo1NS44MTkxNDEyWiIsInVpZCI6MjkyMDI5MiwiZmYiOlsiRmxleGlibGVHcHUiLCJLZXJuZWxzSW50ZXJuZXQiLCJEYXRhRXhwbG9yZXJWMiIsIkRhdGFTb3VyY2VTZWxlY3RvclYyIiwiS2VybmVsc1ZpZXdlcklubmVyVGFibGVPZkNvbnRlbnRzIiwiRm9ydW1XYXRjaERlcHJlY2F0ZWQiLCJOZXdLZXJuZWxXZWxjb21lIiwiTWRlSW1hZ2VVcGxvYWRlciIsIktlcm5lbHNRdWlja1ZlcnNpb25zIiwiRGlzYWJsZUN1c3RvbVBhY2thZ2VzIiwiRG9ja2VyTW9kYWxTZWxlY3RvciIsIlBob25lVmVyaWZ5Rm9yR3B1IiwiQ2xvdWRTZXJ2aWNlc0tlcm5lbEludGVnIiwiVXNlclNlY3JldHNLZXJuZWxJbnRlZyIsIk5hdmlnYXRpb25SZWRlc2lnbiIsIktlcm5lbHNTbmlwcGV0cyIsIktlcm5lbFdlbGNvbWVMb2FkRnJvbVVybCIsIlRwdUtlcm5lbEludGVnIiwiS2VybmVsc0ZpcmViYXNlTG9uZ1BvbGxpbmciLCJEYXRhc2V0TGl2ZU1vdW50IiwiRGF0YXNldHNUYXNrT25Ob3RlYm9va0xpc3RpbmciLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzVHJlZUxlZnQiXSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lEQU5HWEZIdFNJVmM1MU1JZEd3ZzRtUUZnbTNvTnJLb28iLCJibGQiOiJiMTY2NjFjNTM1NWY1OTc0MTJkM2I3YzQxZWQ2ZjJjNzQ4OTNkM2YyIn0.; XSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA; .AspNetCore.Mvc.CookieTempDataProvider=CfDJ8LdUzqlsSWBPr4Ce3rb9VL_w47yY348Q7ujxrJe56Qnw3BV7ZHz6jgCg8lWAdxqhqmV3JunNhR0L6bWgi8helUnYQzK-jUfjNOzpSjjXlDJCkslmO7kT8yUMsWCSqKJyypKsXZO76MTVvnBq_s12ZFo' -H 'origin: https://www.kaggle.com' -H 'x-xsrf-token: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' -H 'accept-language: en-US,en;q=0.8' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Falkon/3.1.0 Chrome/69.0.3497.128 Safari/537.36' -H 'content-type: application/json' -H 'accept: application/json' -H 'referer: https://www.kaggle.com/k1gaggle/jigsaw-multilingual-toxicity-eda-models/edit/run/34120418' -H 'authority: www.kaggle.com' -H 'accept-encoding: gzip, deflate, br' -H '__requestverificationtoken: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' --data-binary '{"kernelRunId":34390363}' --compressed
```

## cancel kernel 

```sh
curl 'https://www.kaggle.com/requests/CreateKernelRunRequest' -H 'cookie: ka_sessionid=cc7e3a02c961799f3f7f281936848379; CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9kJehreeIrmSjmbQiumQNDP6K4pWAAyWOqWypIr_1-3cZm1KXwF4gMCGaMgbMQxLflFuS75NOWQC97EPRiGONyX4INqCDfF7cderxdVyFxUks5x9v6s4580M3CA-gdfYc; GCLB=CNeAmMPxnciELQ; _ga=GA1.2.1299897931.1589986631; _gid=GA1.2.519973049.1589986631; .ASPXAUTH=D4DD912AF0A259BC9170EBC507D60B2B1E932C9443B6D2FF4A604DB1E44DC38810688D6E075771E8D5BCF3CFA60AD0CF8EEC55FA8CABD17FE7B3F49777D80F5A32EE2209FAC1589A00EDBA88A46E17885F256939; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJrMWdhZ2dsZSIsIm5idCI6IjIwMjAtMDUtMjBUMTQ6NTg6NTUuODE5MTQxMloiLCJpYXQiOiIyMDIwLTA1LTIwVDE0OjU4OjU1LjgxOTE0MTJaIiwianRpIjoiNzgyODU5YzctM2YxMS00ZTllLThjOTEtZGNmYTg0NmI0MGNjIiwiZXhwIjoiMjAyMC0wNi0yMFQxNDo1ODo1NS44MTkxNDEyWiIsInVpZCI6MjkyMDI5MiwiZmYiOlsiRmxleGlibGVHcHUiLCJLZXJuZWxzSW50ZXJuZXQiLCJEYXRhRXhwbG9yZXJWMiIsIkRhdGFTb3VyY2VTZWxlY3RvclYyIiwiS2VybmVsc1ZpZXdlcklubmVyVGFibGVPZkNvbnRlbnRzIiwiRm9ydW1XYXRjaERlcHJlY2F0ZWQiLCJOZXdLZXJuZWxXZWxjb21lIiwiTWRlSW1hZ2VVcGxvYWRlciIsIktlcm5lbHNRdWlja1ZlcnNpb25zIiwiRGlzYWJsZUN1c3RvbVBhY2thZ2VzIiwiRG9ja2VyTW9kYWxTZWxlY3RvciIsIlBob25lVmVyaWZ5Rm9yR3B1IiwiQ2xvdWRTZXJ2aWNlc0tlcm5lbEludGVnIiwiVXNlclNlY3JldHNLZXJuZWxJbnRlZyIsIk5hdmlnYXRpb25SZWRlc2lnbiIsIktlcm5lbHNTbmlwcGV0cyIsIktlcm5lbFdlbGNvbWVMb2FkRnJvbVVybCIsIlRwdUtlcm5lbEludGVnIiwiS2VybmVsc0ZpcmViYXNlTG9uZ1BvbGxpbmciLCJEYXRhc2V0TGl2ZU1vdW50IiwiRGF0YXNldHNUYXNrT25Ob3RlYm9va0xpc3RpbmciLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzVHJlZUxlZnQiXSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lEQU5HWEZIdFNJVmM1MU1JZEd3ZzRtUUZnbTNvTnJLb28iLCJibGQiOiJiMTY2NjFjNTM1NWY1OTc0MTJkM2I3YzQxZWQ2ZjJjNzQ4OTNkM2YyIn0.; XSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA; .AspNetCore.Mvc.CookieTempDataProvider=CfDJ8LdUzqlsSWBPr4Ce3rb9VL8TNODiRaawZrt_VDm0CS4zV5Q597_0H9frix7V0RU7VgTBdfbS389NB3fXfkv59SNFrUYSCGNlyaU_S41DIXM4PKJ4-7E1h1R4p1mG0d_j95ig_ydsv0NheRHLB1BXEqc' -H 'origin: https://www.kaggle.com' -H 'x-xsrf-token: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' -H 'accept-language: en-US,en;q=0.8' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Falkon/3.1.0 Chrome/69.0.3497.128 Safari/537.36' -H 'content-type: application/json' -H 'accept: application/json' -H 'referer: https://www.kaggle.com/k1gaggle/jigsaw-multilingual-toxicity-eda-models/edit/run/34120418' -H 'authority: www.kaggle.com' -H 'accept-encoding: gzip, deflate, br' -H '__requestverificationtoken: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' --data-binary '{"forkedFromKernelId":null,"kernelVersionId":null,"dockerImageVersionId":null,"pinnedDockerImageVersionId":null,"ipynbContents":null,"diff":null,"forkParentDiff":null,"versionType":"interactive","kernelId":9350196,"title":"Jigsaw Multilingual Toxicity : EDA + Models 🤬","languageId":8,"dockerImageTag":null,"workerPoolName":null,"dataSources":[{"sourceType":"Competition","sourceId":19018,"databundleVersionId":1053282,"mountSlug":"jigsaw-multilingual-toxic-comment-classification"},{"sourceType":"DatasetVersion","sourceId":1166112,"databundleVersionId":null,"mountSlug":"jigsaw-multilingula-toxicity-token-encoded"}],"compute":{"accelerator":"tpu_v3_8","internet":{"isEnabled":true},"constraints":null},"useGivenDataSourceVersionIds":false}' --compressed
```

```javascript
fetch("https://www.kaggle.com/requests/CreateKernelRunRequest", {"credentials":"include","headers":{},"referrer":"https://www.kaggle.com/k1gaggle/jigsaw-multilingual-toxicity-eda-models/edit/run/34120418","referrerPolicy":"strict-origin-when-cross-origin","body":"{\"forkedFromKernelId\":null,\"kernelVersionId\":null,\"dockerImageVersionId\":null,\"pinnedDockerImageVersionId\":null,\"ipynbContents\":null,\"diff\":null,\"forkParentDiff\":null,\"versionType\":\"interactive\",\"kernelId\":9350196,\"title\":\"Jigsaw Multilingual Toxicity : EDA + Models 🤬\",\"languageId\":8,\"dockerImageTag\":null,\"workerPoolName\":null,\"dataSources\":[{\"sourceType\":\"Competition\",\"sourceId\":19018,\"databundleVersionId\":1053282,\"mountSlug\":\"jigsaw-multilingual-toxic-comment-classification\"},{\"sourceType\":\"DatasetVersion\",\"sourceId\":1166112,\"databundleVersionId\":null,\"mountSlug\":\"jigsaw-multilingula-toxicity-token-encoded\"}],\"compute\":{\"accelerator\":\"tpu_v3_8\",\"internet\":{\"isEnabled\":true},\"constraints\":null},\"useGivenDataSourceVersionIds\":false}","method":"POST","mode":"cors"});
```

## create kernel run request

# multi-language bert
[github link](https://github.com/google-research/bert)

## Fine-tuning Example
[link](https://github.com/google-research/bert)

-> just push the button and let it run
1. get the model + data
2. put it to the computing server (model + data)
3. run...the test

```sh
export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export XNLI_DIR=/path/to/xnli

python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=0.1 \
  --output_dir=/tmp/xnli_output/
```
In the above code, XNLI is the XNLI dataset, whose link (and model/data links) you can find in the [link](https://github.com/google-research/bert)

The BERT\_BASE\_DIR stores vocab, bert\_config, and bert\_model.ckpt

[XNLI dataset, github](https://github.com/facebookresearch/XNLI)

[Bert-Base multilingual case, zip](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)
[XNLI dev/test set, zip](https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip)
[XNLI machine-translated training set, zip](https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip)

```sh
# commmands to prepare dataset to kaggle dataset -> put to kaggle kernel, download and unzip data, then exit
curl -O https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip
curl -O https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip
curl -O https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

extrace_to_folder () {
  zip_name=$1
  zn=$(echo $zip_name | sed 's/\(.*\)\.zip/\1/')
  #mkdir "$zn" && pushd "$zn" && unzip $OLDPWD/"$zip_name" && popd
  unzip -d "$zn" "$zip_name"
}
export -f extrace_to_folder

ls *.zip | xargs -I{} bash -c 'extrace_to_folder {}'

DEV_TEST=XNLI-1.0
MT_TRAIN=XNLI-MT-1.0

mkdir XNLI && mv $DEV_TEST/* XNLI && mv $MT_TRAIN/* XNLI
rmdir DEV_TEST
rmdir MT_TRAIN

# test our datasets
export BERT_BASE_DIR=$PWD/multi_cased_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export XNLI_DIR=$PWD/XNLI

git clone --depth=1 https://github.com/google-research/bert
cd bert && python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=0.1 \
  --output_dir=/tmp/xnli_output/
```

## TODO
1. bert data preprocess, remove "" at heads and tails, update datasets
1. split module to prepare datasets
