# APIs
## cancel kernel 

```sh
curl 'https://www.kaggle.com/requests/CreateKernelRunRequest' -H 'cookie: ka_sessionid=cc7e3a02c961799f3f7f281936848379; CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9kJehreeIrmSjmbQiumQNDP6K4pWAAyWOqWypIr_1-3cZm1KXwF4gMCGaMgbMQxLflFuS75NOWQC97EPRiGONyX4INqCDfF7cderxdVyFxUks5x9v6s4580M3CA-gdfYc; GCLB=CNeAmMPxnciELQ; _ga=GA1.2.1299897931.1589986631; _gid=GA1.2.519973049.1589986631; .ASPXAUTH=D4DD912AF0A259BC9170EBC507D60B2B1E932C9443B6D2FF4A604DB1E44DC38810688D6E075771E8D5BCF3CFA60AD0CF8EEC55FA8CABD17FE7B3F49777D80F5A32EE2209FAC1589A00EDBA88A46E17885F256939; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJrMWdhZ2dsZSIsIm5idCI6IjIwMjAtMDUtMjBUMTQ6NTg6NTUuODE5MTQxMloiLCJpYXQiOiIyMDIwLTA1LTIwVDE0OjU4OjU1LjgxOTE0MTJaIiwianRpIjoiNzgyODU5YzctM2YxMS00ZTllLThjOTEtZGNmYTg0NmI0MGNjIiwiZXhwIjoiMjAyMC0wNi0yMFQxNDo1ODo1NS44MTkxNDEyWiIsInVpZCI6MjkyMDI5MiwiZmYiOlsiRmxleGlibGVHcHUiLCJLZXJuZWxzSW50ZXJuZXQiLCJEYXRhRXhwbG9yZXJWMiIsIkRhdGFTb3VyY2VTZWxlY3RvclYyIiwiS2VybmVsc1ZpZXdlcklubmVyVGFibGVPZkNvbnRlbnRzIiwiRm9ydW1XYXRjaERlcHJlY2F0ZWQiLCJOZXdLZXJuZWxXZWxjb21lIiwiTWRlSW1hZ2VVcGxvYWRlciIsIktlcm5lbHNRdWlja1ZlcnNpb25zIiwiRGlzYWJsZUN1c3RvbVBhY2thZ2VzIiwiRG9ja2VyTW9kYWxTZWxlY3RvciIsIlBob25lVmVyaWZ5Rm9yR3B1IiwiQ2xvdWRTZXJ2aWNlc0tlcm5lbEludGVnIiwiVXNlclNlY3JldHNLZXJuZWxJbnRlZyIsIk5hdmlnYXRpb25SZWRlc2lnbiIsIktlcm5lbHNTbmlwcGV0cyIsIktlcm5lbFdlbGNvbWVMb2FkRnJvbVVybCIsIlRwdUtlcm5lbEludGVnIiwiS2VybmVsc0ZpcmViYXNlTG9uZ1BvbGxpbmciLCJEYXRhc2V0TGl2ZU1vdW50IiwiRGF0YXNldHNUYXNrT25Ob3RlYm9va0xpc3RpbmciLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzVHJlZUxlZnQiXSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lEQU5HWEZIdFNJVmM1MU1JZEd3ZzRtUUZnbTNvTnJLb28iLCJibGQiOiJiMTY2NjFjNTM1NWY1OTc0MTJkM2I3YzQxZWQ2ZjJjNzQ4OTNkM2YyIn0.; XSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA; .AspNetCore.Mvc.CookieTempDataProvider=CfDJ8LdUzqlsSWBPr4Ce3rb9VL8TNODiRaawZrt_VDm0CS4zV5Q597_0H9frix7V0RU7VgTBdfbS389NB3fXfkv59SNFrUYSCGNlyaU_S41DIXM4PKJ4-7E1h1R4p1mG0d_j95ig_ydsv0NheRHLB1BXEqc' -H 'origin: https://www.kaggle.com' -H 'x-xsrf-token: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' -H 'accept-language: en-US,en;q=0.8' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Falkon/3.1.0 Chrome/69.0.3497.128 Safari/537.36' -H 'content-type: application/json' -H 'accept: application/json' -H 'referer: https://www.kaggle.com/k1gaggle/jigsaw-multilingual-toxicity-eda-models/edit/run/34120418' -H 'authority: www.kaggle.com' -H 'accept-encoding: gzip, deflate, br' -H '__requestverificationtoken: CfDJ8LdUzqlsSWBPr4Ce3rb9VL9QRaVweKSwzfKrcmHPaodsLOGCBWIjmBd2-h7HKDq9tslwXR9F5sz8BsUBEG3GOu1VpKRsZ2BSsaT4L2-qdgL8k2JOYFEkzqUjRgKY8dpFrUq-ini3CuV4CUwNmtn2BhJBCJBI5Lf2Z-oZPy1SI04MqJ7edzcy3GaGvgAbg92RHA' --data-binary '{"forkedFromKernelId":null,"kernelVersionId":null,"dockerImageVersionId":null,"pinnedDockerImageVersionId":null,"ipynbContents":null,"diff":null,"forkParentDiff":null,"versionType":"interactive","kernelId":9350196,"title":"Jigsaw Multilingual Toxicity : EDA + Models 🤬","languageId":8,"dockerImageTag":null,"workerPoolName":null,"dataSources":[{"sourceType":"Competition","sourceId":19018,"databundleVersionId":1053282,"mountSlug":"jigsaw-multilingual-toxic-comment-classification"},{"sourceType":"DatasetVersion","sourceId":1166112,"databundleVersionId":null,"mountSlug":"jigsaw-multilingula-toxicity-token-encoded"}],"compute":{"accelerator":"tpu_v3_8","internet":{"isEnabled":true},"constraints":null},"useGivenDataSourceVersionIds":false}' --compressed
```

```javascript
fetch("https://www.kaggle.com/requests/CreateKernelRunRequest", {"credentials":"include","headers":{},"referrer":"https://www.kaggle.com/k1gaggle/jigsaw-multilingual-toxicity-eda-models/edit/run/34120418","referrerPolicy":"strict-origin-when-cross-origin","body":"{\"forkedFromKernelId\":null,\"kernelVersionId\":null,\"dockerImageVersionId\":null,\"pinnedDockerImageVersionId\":null,\"ipynbContents\":null,\"diff\":null,\"forkParentDiff\":null,\"versionType\":\"interactive\",\"kernelId\":9350196,\"title\":\"Jigsaw Multilingual Toxicity : EDA + Models 🤬\",\"languageId\":8,\"dockerImageTag\":null,\"workerPoolName\":null,\"dataSources\":[{\"sourceType\":\"Competition\",\"sourceId\":19018,\"databundleVersionId\":1053282,\"mountSlug\":\"jigsaw-multilingual-toxic-comment-classification\"},{\"sourceType\":\"DatasetVersion\",\"sourceId\":1166112,\"databundleVersionId\":null,\"mountSlug\":\"jigsaw-multilingula-toxicity-token-encoded\"}],\"compute\":{\"accelerator\":\"tpu_v3_8\",\"internet\":{\"isEnabled\":true},\"constraints\":null},\"useGivenDataSourceVersionIds\":false}","method":"POST","mode":"cors"});
```

## create kernel run request
SOMETHING

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
  --max\_seq_length=128 \
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
  --max\_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=0.1 \
  --output_dir=/tmp/xnli_output/
```

## TODO
1. bert data pre-process, remove "" at heads and tails, update datasets
1. split module to prepare datasets

# submission log

## 0610: version 6, 
from kaggle_runner import may_debug

class LabelSmoothing(nn.Module):
    """https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631"""

    def __init__(self, smoothing = 0.1, dim=-1):
        super(LabelSmoothing, self).__init__()
        self.cls = 2
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, x, target):
        may_debug()
        if self.training:
            pred = x[:,:2].log_softmax(dim=self.dim)
            aux=x[:, 2:]

            toxic_target = target[:,:2]
            aux\_target = target[:, 2:]
            with torch.no_grad():
                # true_dist = pred.data.clone()
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing) # hardcode for binary classification
                true_dist += (1-self.smoothing*2)*toxic_target
                # true_dist.scatter_(1, toxic_target.data.unsqueeze(1), self.confidence) # only for 0 1 label, put confidence to related place

                # for 0-1, 0 -> 0.1, 1->0.9.(if 1), if zero. 0->0.9, 1->0.1
                smooth_aux = torch.zeros_like(aux\_target)
                smooth_aux.fill_(self.smoothing) # only for binary cross entropy
                smooth_aux += (1-self.smoothing*2)*aux\_target  # only for binary cross entropy, so for lable, it is (1-smooth)*

            aux\_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux, smooth_aux)

            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) + aux\_loss/5
        else:
            return torch.nn.functional.binary_cross_entropy_with_logits(x[:,:2], target[:,:2])

class TrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs
    n_epochs = 2  # количество эпох для обучения
    lr = 0.5 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
    fold_number = 0  # номер фолда для обучения

    # -------------------
    verbose = True  # выводить принты
    verbose_step = 50  # количество шагов для вывода принта
    # -------------------

    # --------------------
    step_scheduler = False  # выполнять scheduler.step после вызова optimizer.step
    validation_scheduler = True  # выполнять scheduler.step после валидации loss (например для плато)
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # -------------------
    criterion = LabelSmoothing()
    # -------------------

```
Train Step 2300, loss: 0.46885, final_score: 0.98592, time: 2692.24801
[RESULT]: Train. Epoch: 1, loss: 0.46894, final_score: 0.98586, time: 2737.68178
[RESULT]: Validation. Epoch: 1, loss: 0.63544, final_score: 0.95572, time: 63.45235
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.07880
Train Step 50, loss: 0.48480, final_score: 0.94857, time: 119.47729
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.07922
Train Step 50, loss: 0.46219, final_score: 0.96771, time: 60.06645
```
**score**  0.9421
**change** use aux

### Version 7
3 epochs, and for aux it is /3
```

Train Step 2300, loss: 0.51410, final_score: 0.98812, time: 2806.45758
[RESULT]: Train. Epoch: 2, loss: 0.51399, final_score: 0.98818, time: 2852.04364
[RESULT]: Validation. Epoch: 2, loss: 0.65034, final_score: 0.95346, time: 66.14816
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.09183
Train Step 50, loss: 0.53016, final_score: 0.94703, time: 59.50609
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.09470
Train Step 50, loss: 0.50945, final_score: 0.96875, time: 60.62506

```

*change* 3 epochs, aux /3
**score** 0.9431 (merge with version 6 results so better)

#### compare logs
as we trained on validation set, it will always get better.
train loss less(longer) and the final test result is better

### version 8
error for providing test datasets

### version 9
save the model after using more data augmentation

### Version 10
test dummy submission
**score** 0.8897

### Version 11
*change* more data augmentation
might be over-fit or some other problem
3 epochs
```
[RESULT]: Train. Epoch: 2, loss: 0.50794, final_score: 0.98926, time: 2459.93479
[RESULT]: Validation. Epoch: 2, loss: 0.73969, final_score: 0.94894, time: 28.61954
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10969
Train Step 25, loss: 0.53102, final_score: 0.97023, time: 89.43301
Train Step 50, loss: 0.52637, final_score: 0.97381, time: 115.44825
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10646
Train Step 25, loss: 0.50914, final_score: 0.98519, time: 26.09558
Train Step 50, loss: 0.50297, final_score: 0.98700, time: 52.20791
```
**score** 0.9310

## Version 12
test failed again (no submission.csv), 3 epochs, data augmentation no mix

```
[RESULT]: Train. Epoch: 2, loss: 0.50892, final_score: 0.99015, time: 2470.52454
[RESULT]: Validation. Epoch: 2, loss: 0.64159, final_score: 0.95372, time:
28.67925
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.11701
Train Step 25, loss: 0.51453, final_score: 0.96150, time: 26.66932
Train Step 50, loss: 0.51703, final_score: 0.95678, time: 52.89321
```

## **Version 13**

test 1 epoch, data aug no mix
```
[RESULT]: Train. Epoch: 0, loss: 0.53342, final_score: 0.97880, time: 2472.48243
[RESULT]: Validation. Epoch: 0, loss: 0.62269, final_score: 0.95234, time: 29.54635
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10594
Train Step 25, loss: 0.50945, final_score: 0.96992, time: 25.95953
Train Step 50, loss: 0.51980, final_score: 0.95819, time: 52.14686
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10220
Train Step 25, loss: 0.49641, final_score: 0.98152, time: 26.33078
Train Step 50, loss: 0.49699, final_score: 0.97833, time: 52.59766
```
**score** 0.9411

## Vesion 14

2 epochs, data aug no mix
```
[RESULT]: Train. Epoch: 0, loss: 0.53821, final_score: 0.97950, time: 2481.05183
[RESULT]: Validation. Epoch: 0, loss: 0.59097, final_score: 0.95094, time: 29.88211

[RESULT]: Train. Epoch: 1, loss: 0.51774, final_score: 0.98717, time: 2470.12558
[RESULT]: Validation. Epoch: 1, loss: 0.61562, final_score: 0.95022, time: 28.81456
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.11438
Train Step 25, loss: 0.52422, final_score: 0.93982, time: 26.49946
Train Step 50, loss: 0.52055, final_score: 0.94838, time: 52.72035
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10594
Train Step 25, loss: 0.49531, final_score: 0.98055, time: 26.42729
Train Step 50, loss: 0.50309, final_score: 0.97200, time: 52.56680
```

**score** 0.9423
## Version 15
pickle data

## Version 16
aux\_loss/1, 3 epochs

```
[RESULT]: Train. Epoch: 2, loss: 0.76647, final_score: 0.99029, time: 2454.93416
[RESULT]: Validation. Epoch: 2, loss: 0.95294, final_score: 0.95030, time: 28.61084
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.09192
Train Step 25, loss: 0.73516, final_score: 0.95130, time: 89.40943
Train Step 50, loss: 0.73148, final_score: 0.96084, time: 115.37684
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.09284
Train Step 25, loss: 0.71062, final_score: 0.98202, time: 25.92905
Train Step 50, loss: 0.71711, final_score: 0.97655, time: 51.97299
```
**score** 0.9393

Analyze: maybe the toxic data too much, easier to over-fit

## Version 17

aux\_loss/3, 3 epochs

```
[RESULT]: Train. Epoch: 2, loss: 0.51123, final_score: 0.98982, time: 2457.49771
[RESULT]: Validation. Epoch: 2, loss: 0.62919, final_score: 0.95049, time: 28.64095
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.10021
Train Step 25, loss: 0.51969, final_score: 0.95565, time: 89.48005
Train Step 50, loss: 0.51586, final_score: 0.95909, time: 115.79362
Train Step 0, loss: 0.00000, final_score: 0.00000, time: 0.09716
Train Step 25, loss: 0.48781, final_score: 0.98629, time: 26.15233
Train Step 50, loss: 0.49199, final_score: 0.98348, time: 52.17845
```
**score** 0.9409

## Version 18
aux\_loss/3, 3 epochs, no toxic data to synthesizer, but merge more data with synthe
```
[RESULT]: Train. Epoch: 2, loss: 0.51623, final_score: 0.98815, mc_score: 0.89592, time: 2479.38604
[RESULT]: Validation. Epoch: 2, loss: 0.67237, final_score: 0.95114, mc_score: 0.56823, time: 28.99195
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.11007
Train Step 25, loss: 0.52273, final_score: 0.94748, mc_score: 0.72564, time: 89.66919
Train Step 50, loss: 0.52945, final_score: 0.94235, mc_score: 0.69355, time: 116.05590
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.10153
Train Step 25, loss: 0.49320, final_score: 0.98264, mc_score: 0.77971, time: 26.53590
Train Step 50, loss: 0.49980, final_score: 0.97090, mc_score: 0.78187, time: 53.01024
```

**score** 0.9386

## Version 19

aux\_loss/3, 2 epochs
not \_low syn to data. And reduce label smooth

```
[RESULT]: Train. Epoch: 1, loss: 0.38311, final_score: 0.98706, mc_score: 0.89245, time: 2484.84076
[RESULT]: Validation. Epoch: 1, loss: 0.49391, final_score: 0.95422, mc_score: 0.61094, time: 29.02337
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.09500
Train Step 25, loss: 0.39242, final_score: 0.95024, mc_score: 0.67116, time: 91.17837
Train Step 50, loss: 0.40027, final_score: 0.94823, mc_score: 0.66167, time: 117.53904
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.09492
Train Step 25, loss: 0.35906, final_score: 0.97616, mc_score: 0.72517, time: 26.31303
Train Step 50, loss: 0.36391, final_score: 0.96935, mc_score: 0.75350, time: 52.64425
```
**score** 0.9393

## Version 20

one more epoch(or just one epoch?)-> just one epoch, but result combined with more?

```
[RESULT]: Train. Epoch: 0, loss: 0.55424, final_score: 0.97425, mc_score: 0.83946, time: 2483.88474
[RESULT]: Validation. Epoch: 0, loss: 0.61969, final_score: 0.95411, mc_score: 0.58527, time: 29.85189
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.09249
Train Step 25, loss: 0.53078, final_score: 0.94804, mc_score: 0.66260, time: 26.12033
Train Step 50, loss: 0.52883, final_score: 0.94932, mc_score: 0.71221, time: 52.60766
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.10790
Train Step 25, loss: 0.51930, final_score: 0.96550, mc_score: 0.65855, time: 26.18452
Train Step 50, loss: 0.51625, final_score: 0.96210, mc_score: 0.72319, time: 52.60593
```
**score** 0.9418

## Version 21, back to smooth 0.1
```
[RESULT]: Train. Epoch: 2, loss: 0.53283, final_score: 0.98131, mc_score: 0.89359, time: 2993.50178
[RESULT]: Validation. Epoch: 2, loss: 0.63669, final_score: 0.94887, mc_score: 0.58527, time: 28.16652
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.08968
Train Step 25, loss: 0.52883, final_score: 0.95406, mc_score: 0.72143, time: 117.51615
Train Step 50, loss: 0.53277, final_score: 0.94861, mc_score: 0.70394, time: 149.42057
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.09337
Train Step 25, loss: 0.51328, final_score: 0.96808, mc_score: 0.77331, time: 31.67108
Train Step 50, loss: 0.51551, final_score: 0.96593, mc_score: 0.76808, time: 63.67104
```
**score** 0.9408

## Version 22
smooth to 0.05
just test one batch (with out merging others), data normal with \_low
synthesizer
```
[RESULT]: Train. Epoch: 0, loss: 0.42999, final_score: 0.97412, mc_score: 0.85195, time: 3204.96820
[RESULT]: Validation. Epoch: 0, loss: 0.50314, final_score: 0.95609, mc_score: 0.59917, time: 28.97322
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.09731
Train Step 25, loss: 0.39695, final_score: 0.95353, mc_score: 0.72228, time: 31.38654
Train Step 50, loss: 0.40520, final_score: 0.94530, mc_score: 0.69190, time: 63.66373
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.08831
Train Step 25, loss: 0.38891, final_score: 0.95604, mc_score: 0.74218, time: 31.40544
Train Step 50, loss: 0.39375, final_score: 0.95146, mc_score: 0.73824, time: 63.41240
```
**score** 0.9360

# Merging just makes it better

V22+V21-> 0.9393

## Version 23

data normal no synthesizer (merge with previous results)
and aux\_loss /5
```
[RESULT]: Train. Epoch: 1, loss: 0.49240, final_score: 0.97706, mc_score: 0.87177, time: 3006.81131
[RESULT]: Validation. Epoch: 1, loss: 0.59272, final_score: 0.95454, mc_score: 0.60239, time: 28.31951
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.08864
Train Step 25, loss: 0.49711, final_score: 0.93300, mc_score: 0.69818, time: 32.03840
Train Step 50, loss: 0.50258, final_score: 0.92377, mc_score: 0.69327, time: 64.11194
Train Step 0, loss: 0.00000, final_score: 0.00000, mc_score: 0.00000, time: 0.08858
Train Step 25, loss: 0.47227, final_score: 0.95919, mc_score: 0.74816, time: 31.59596
Train Step 50, loss: 0.48590, final_score: 0.94189, mc_score: 0.71498, time: 63.50205
```

**score** 0.9429

[Train hooks](https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/basic_train.py#L85)
[backend callback](https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/callback.py#L297)
[Step Update parameters](https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/general_optimizer.py#L97)
So we can step on self.learn.opt.on_step to check the grad.data check if it is almost the same range/abs


```
Train Step 1, loss: 1.10938, final_score: 0.51562, mc_score: 0.37796, time: 113.06641
Train Step 2, loss: 1.10938, final_score: 0.53137, mc_score: 0.05929, time: 204.88724
Train Step 3, loss: 1.18490, final_score: 0.46786, mc_score: -0.07559, time: 206.35083
Train Step 4, loss: 1.16992, final_score: 0.42993, mc_score: -0.17491, time: 207.75350
Train Step 5, loss: 1.13359, final_score: 0.41672, mc_score: -0.21288, time: 209.00740
Train Step 6, loss: 1.13086, final_score: 0.39510, mc_score: -0.23869, time: 210.28957
Train Step 7, loss: 1.10212, final_score: 0.40630, mc_score: -0.20868, time: 211.59308
Train Step 8, loss: 1.11377, final_score: 0.39846, mc_score: -0.18700, time: 212.86807
Train Step 9, loss: 1.08507, final_score: 0.44333, mc_score: -0.13420, time: 214.13142
Train Step 10, loss: 1.06992, final_score: 0.44571, mc_score: -0.11746, time: 215.47282
Train Step 11, loss: 1.05575, final_score: 0.45519, mc_score: -0.10518, time: 216.76567
Train Step 12, loss: 1.04167, final_score: 0.46181, mc_score: -0.07926, time: 218.07405
Train Step 13, loss: 1.03906, final_score: 0.45821, mc_score: -0.08050, time: 219.39437
Train Step 14, loss: 1.03404, final_score: 0.46011, mc_score: -0.07815, time: 220.69584
Train Step 15, loss: 1.03385, final_score: 0.45526, mc_score: -0.07927, time: 222.01984
Train Step 16, loss: 1.02637, final_score: 0.45979, mc_score: -0.07169, time: 223.34185
Train Step 17, loss: 1.02114, final_score: 0.46996, mc_score: -0.06559, time: 224.66254
Train Step 18, loss: 1.02344, final_score: 0.45818, mc_score: -0.08446, time: 225.97743
Train Step 19, loss: 1.01295, final_score: 0.46861, mc_score: -0.06522, time: 227.28586
Train Step 20, loss: 1.01133, final_score: 0.47783, mc_score: -0.05318, time: 228.60961
Train Step 21, loss: 0.99851, final_score: 0.50032, mc_score: -0.02070, time: 229.89782
Train Step 22, loss: 0.99219, final_score: 0.51287, mc_score: 0.00434, time: 231.16871
Train Step 23, loss: 0.98285, final_score: 0.52255, mc_score: 0.01971, time: 232.47061
Train Step 24, loss: 0.97461, final_score: 0.53386, mc_score: 0.02858, time: 233.74266
Train Step 25, loss: 0.96531, final_score: 0.54557, mc_score: 0.04745, time: 235.05073
Train Step 26, loss: 0.95733, final_score: 0.55457, mc_score: 0.06554, time: 236.32955
Train Step 27, loss: 0.94589, final_score: 0.57111, mc_score: 0.09263, time: 237.59322
Train Step 28, loss: 0.93443, final_score: 0.58872, mc_score: 0.11715, time: 238.92549
Train Step 29, loss: 0.92659, final_score: 0.59963, mc_score: 0.13013, time: 240.18953
Train Step 30, loss: 0.92201, final_score: 0.60802, mc_score: 0.13284, time: 241.45309
Train Step 31, loss: 0.91179, final_score: 0.62168, mc_score: 0.14895, time: 242.71853
Train Step 32, loss: 0.90149, final_score: 0.63606, mc_score: 0.16827, time: 243.98781
Train Step 33, loss: 0.89607, final_score: 0.64571, mc_score: 0.18317, time: 245.31105
Train Step 34, loss: 0.89223, final_score: 0.65074, mc_score: 0.18890, time: 246.63058
Train Step 35, loss: 0.88906, final_score: 0.65847, mc_score: 0.20050, time: 247.88921
Train Step 36, loss: 0.88411, final_score: 0.67138, mc_score: 0.21973, time: 249.15814
Train Step 37, loss: 0.87679, final_score: 0.67997, mc_score: 0.23108, time: 250.47610
Train Step 38, loss: 0.87130, final_score: 0.68700, mc_score: 0.24207, time: 251.79559
Train Step 39, loss: 0.86729, final_score: 0.69427, mc_score: 0.25179, time: 253.11523
Train Step 40, loss: 0.86006, final_score: 0.70424, mc_score: 0.26366, time: 254.44200
Train Step 41, loss: 0.85147, final_score: 0.71584, mc_score: 0.28186, time: 255.70692
Train Step 42, loss: 0.84756, final_score: 0.72371, mc_score: 0.29279, time: 256.97036
Train Step 43, loss: 0.84366, final_score: 0.73116, mc_score: 0.30665, time: 258.28103
Train Step 44, loss: 0.83727, final_score: 0.73851, mc_score: 0.31621, time: 259.53896
Train Step 45, loss: 0.83472, final_score: 0.74288, mc_score: 0.32581, time: 260.78796
Train Step 46, loss: 0.83076, final_score: 0.74845, mc_score: 0.33520, time: 262.05888
Train Step 47, loss: 0.82605, final_score: 0.75570, mc_score: 0.34660, time: 263.32842
Train Step 48, loss: 0.81925, final_score: 0.76441, mc_score: 0.35966, time: 264.58692
Train Step 49, loss: 0.81449, final_score: 0.77046, mc_score: 0.36724, time: 265.88840
Train Step 50, loss: 0.81398, final_score: 0.77245, mc_score: 0.37032, time: 267.18382
Train Step 51, loss: 0.81005, final_score: 0.77770, mc_score: 0.37730, time: 268.45383
Train Step 52, loss: 0.80424, final_score: 0.78382, mc_score: 0.38800, time: 269.74650
Train Step 53, loss: 0.80085, final_score: 0.78749, mc_score: 0.39708, time: 271.11040
Train Step 54, loss: 0.79615, final_score: 0.79201, mc_score: 0.40578, time: 272.36462
```
The first two batches, 100ms each. About 1 second one batch.

The following CPU, data not downsample
```  
[DEBUG]2020-06-18 10:36:24,981:utils:Device used: cpu
[DEBUG]2020-06-18 10:36:25,041:utils:Train Step 0, loss: 0.00000, lr: 5e-06 final_score: 0.00000, mc_score: 0.00000, time: 0.04135
[DEBUG]2020-06-18 10:36:46,579:utils:Train Step 1, loss: 0.95281, lr: 5e-06 final_score: 0.43590, mc_score: 0.17949, time: 21.57958
[DEBUG]2020-06-18 10:36:59,812:utils:Train Step 2, loss: 0.92699, lr: 5e-06 final_score: 0.29630, mc_score: 0.09759, time: 34.81184
[DEBUG]2020-06-18 10:37:12,832:utils:Train Step 3, loss: 0.81862, lr: 5e-06 final_score: 0.33023, mc_score: 0.14394, time: 47.83250
[DEBUG]2020-06-18 10:37:25,925:utils:Train Step 4, loss: 0.76314, lr: 5e-06 final_score: 0.35593, mc_score: 0.13220, time: 60.92473
[DEBUG]2020-06-18 10:37:38,717:utils:Train Step 5, loss: 0.72612, lr: 5e-06 final_score: 0.38933, mc_score: 0.14667, time: 73.71680
[DEBUG]2020-06-18 10:37:51,702:utils:Train Step 6, loss: 0.72418, lr: 5e-06 final_score: 0.36111, mc_score: 0.13315, time: 86.70245
[DEBUG]2020-06-18 10:38:04,516:utils:Train Step 7, loss: 0.71632, lr: 5e-06 final_score: 0.33197, mc_score: 0.12279, time: 99.51615
[DEBUG]2020-06-18 10:38:17,705:utils:Train Step 8, loss: 0.69501, lr: 5e-06 final_score: 0.34238, mc_score: 0.12886, time: 112.70531
[DEBUG]2020-06-18 10:38:30,389:utils:Train Step 9, loss: 0.68046, lr: 5e-06 final_score: 0.38373, mc_score: 0.13351, time: 125.38881
[DEBUG]2020-06-18 10:38:43,086:utils:Train Step 10, loss: 0.66373, lr: 5e-06 final_score: 0.39402, mc_score: 0.13720, time: 138.08598
[DEBUG]2020-06-18 10:38:55,723:utils:Train Step 11, loss: 0.65396, lr: 5e-06 final_score: 0.47917, mc_score: 0.25964, time: 150.72317
[DEBUG]2020-06-18 10:39:08,344:utils:Train Step 12, loss: 0.64371, lr: 5e-06 final_score: 0.50340, mc_score: 0.26215, time: 163.34438
```

```
Train Step 0, loss: 0.00000, lr: 5e-06 final_score: 0.00000, mc_score: 0.00000, time: 0.01049
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(5.306246) mean(3.284097)
params info pg1: norm std(10.843104) mean(8.226689)
grad info pg1: norm std(0.296042) mean(0.221219)
Train Step 1, loss: 0.82892, lr: 5e-06 final_score: 0.35897, mc_score: 0.00000, time: 12.35283
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(1.390194) mean(0.796119)
params info pg1: norm std(10.843107) mean(8.226690)
grad info pg1: norm std(0.038992) mean(0.048181)
Train Step 2, loss: 0.73110, lr: 5e-06 final_score: 0.57037, mc_score: 0.00000, time: 28.22387
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(0.270485) mean(0.182517)
params info pg1: norm std(10.843109) mean(8.226691)
grad info pg1: norm std(0.009722) mean(0.014251)
Train Step 3, loss: 0.65143, lr: 5e-06 final_score: 0.60465, mc_score: 0.00000, time: 44.68827
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(0.329400) mean(0.266966)
params info pg1: norm std(10.843109) mean(8.226692)
grad info pg1: norm std(0.015222) mean(0.019330)
Train Step 4, loss: 0.60733, lr: 5e-06 final_score: 0.62712, mc_score: 0.00000, time: 59.50831
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(0.934494) mean(0.587859)
params info pg1: norm std(10.843109) mean(8.226692)
grad info pg1: norm std(0.057466) mean(0.042114)
Train Step 5, loss: 0.58537, lr: 5e-06 final_score: 0.62400, mc_score: 0.00000, time: 74.23315
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(0.441422) mean(0.342927)
params info pg1: norm std(10.843109) mean(8.226693)
grad info pg1: norm std(0.025181) mean(0.025235)
Train Step 6, loss: 0.57657, lr: 5e-06 final_score: 0.66667, mc_score: 0.00000, time: 89.13756
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284134)
grad info pg0: norm std(0.335821) mean(0.317519)
params info pg1: norm std(10.843109) mean(8.226694)
grad info pg1: norm std(0.014155) mean(0.023538)
Train Step 7, loss: 0.57544, lr: 5e-06 final_score: 0.68844, mc_score: 0.00000, time: 103.90428
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284142)
grad info pg0: norm std(0.589146) mean(0.350190)
params info pg1: norm std(10.843110) mean(8.226694)
grad info pg1: norm std(0.020629) mean(0.025898)
Train Step 8, loss: 0.56693, lr: 5e-06 final_score: 0.67060, mc_score: 0.00000, time: 118.74932
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284142)
grad info pg0: norm std(0.488801) mean(0.318388)
params info pg1: norm std(10.843110) mean(8.226694)
grad info pg1: norm std(0.017937) mean(0.024180)
Train Step 9, loss: 0.55881, lr: 5e-06 final_score: 0.66215, mc_score: 0.00000, time: 133.70876
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284142)
grad info pg0: norm std(0.846748) mean(0.341456)
params info pg1: norm std(10.843110) mean(8.226694)
grad info pg1: norm std(0.025591) mean(0.026805)
Train Step 10, loss: 0.55699, lr: 5e-06 final_score: 0.64332, mc_score: -0.01696, time: 148.50977
grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
params info pg0: norm std(212.603531) mean(76.284142)
grad info pg0: norm std(0.717754) mean(0.514715)
params info pg1: norm std(10.843110) mean(8.226694)
grad info pg1: norm std(0.035379) mean(0.036088)
Train Step 11, loss: 0.56302, lr: 5e-06 final_score: 0.60565, mc_score: -0.02340, time: 163.13020
...
[DEBUG]2020-06-18 10:47:10,491:utils:grad info: AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.001

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 5e-06
    weight_decay: 0.0
)
[DEBUG]2020-06-18 10:47:10,591:utils:params info pg0: norm std(212.603546) mean(76.284157)
[DEBUG]2020-06-18 10:47:10,593:utils:grad info pg0: norm std(0.693694) mean(0.624429)
[DEBUG]2020-06-18 10:47:10,601:utils:params info pg1: norm std(10.843110) mean(8.226697)
[DEBUG]2020-06-18 10:47:10,603:utils:grad info pg1: norm std(0.031085) mean(0.045465)
[DEBUG]2020-06-18 10:47:11,282:utils:Train Step 22, loss: 0.56390, lr: 5e-06 final_score: 0.52226, mc_score: -0.01595, time: 325.95086
```
```
[DEBUG]2020-06-18 12:58:40,633:utils:Train Step 0, bs: 16, loss: 0.00000, lr: 4e-05 final_score: 0.00000, mc_score: 0.00000, time: 0.46691
[DEBUG]2020-06-18 12:58:54,199:utils:params info pg0: norm std(212.533707) mean(76.280579)
[DEBUG]2020-06-18 12:58:54,201:utils:grad info pg0: norm std(1.955211) mean(0.262538)
[DEBUG]2020-06-18 12:58:54,208:utils:params info pg1: norm std(10.842737) mean(8.226854)
[DEBUG]2020-06-18 12:58:54,210:utils:grad info pg1: norm std(0.047125) mean(0.011311)
[DEBUG]2020-06-18 12:58:55,653:utils:Train Step 1, bs: 16, loss: 1.37878, lr: 4e-05 final_score: 0.18333, mc_score: 0.00000, time: 15.48734
[DEBUG]2020-06-18 12:59:11,001:utils:params info pg0: norm std(212.533722) mean(76.280579)
[DEBUG]2020-06-18 12:59:11,003:utils:grad info pg0: norm std(1.619439) mean(0.206290)
[DEBUG]2020-06-18 12:59:11,010:utils:params info pg1: norm std(10.842738) mean(8.226848)
[DEBUG]2020-06-18 12:59:11,012:utils:grad info pg1: norm std(0.034206) mean(0.008209)
[DEBUG]2020-06-18 12:59:12,198:utils:Train Step 2, bs: 16, loss: 1.13720, lr: 4e-05 final_score: 0.48583, mc_score: 0.21713, time: 32.03194
[DEBUG]2020-06-18 12:59:27,477:utils:params info pg0: norm std(212.533722) mean(76.280586)
[DEBUG]2020-06-18 12:59:27,479:utils:grad info pg0: norm std(0.420249) mean(0.062736)
[DEBUG]2020-06-18 12:59:27,486:utils:params info pg1: norm std(10.842747) mean(8.226842)
[DEBUG]2020-06-18 12:59:27,488:utils:grad info pg1: norm std(0.010296) mean(0.002830)
[DEBUG]2020-06-18 12:59:28,672:utils:Train Step 3, bs: 16, loss: 1.02713, lr: 4e-05 final_score: 0.56614, mc_score: 0.23054, time: 48.50635
[DEBUG]2020-06-18 12:59:44,011:utils:params info pg0: norm std(212.533737) mean(76.280602)
[DEBUG]2020-06-18 12:59:44,013:utils:grad info pg0: norm std(0.611604) mean(0.111733)
[DEBUG]2020-06-18 12:59:44,019:utils:params info pg1: norm std(10.842756) mean(8.226837)
[DEBUG]2020-06-18 12:59:44,021:utils:grad info pg1: norm std(0.031941) mean(0.008205)
[DEBUG]2020-06-18 12:59:45,226:utils:Train Step 4, bs: 16, loss: 0.99862, lr: 4e-05 final_score: 0.60392, mc_score: 0.24309, time: 65.05992
[DEBUG]2020-06-18 13:00:00,721:utils:params info pg0: norm std(212.533737) mean(76.280609)
[DEBUG]2020-06-18 13:00:00,723:utils:grad info pg0: norm std(1.530189) mean(0.219328)
[DEBUG]2020-06-18 13:00:00,730:utils:params info pg1: norm std(10.842748) mean(8.226829)
[DEBUG]2020-06-18 13:00:00,731:utils:grad info pg1: norm std(0.043965) mean(0.011104)
[DEBUG]2020-06-18 13:00:01,939:utils:Train Step 5, bs: 16, loss: 1.00642, lr: 4e-05 final_score: 0.60025, mc_score: 0.19799, time: 81.77277
[DEBUG]2020-06-18 13:00:17,374:utils:params info pg0: norm std(212.533722) mean(76.280624)
[DEBUG]2020-06-18 13:00:17,376:utils:grad info pg0: norm std(0.716371) mean(0.096198)
[DEBUG]2020-06-18 13:00:17,383:utils:params info pg1: norm std(10.842729) mean(8.226818)
[DEBUG]2020-06-18 13:00:17,385:utils:grad info pg1: norm std(0.018238) mean(0.004255)
[DEBUG]2020-06-18 13:00:18,581:utils:Train Step 6, bs: 16, loss: 1.01394, lr: 4e-05 final_score: 0.57986, mc_score: 0.16725, time: 98.41493
[DEBUG]2020-06-18 13:00:33,880:utils:params info pg0: norm std(212.533722) mean(76.280632)
[DEBUG]2020-06-18 13:00:33,882:utils:grad info pg0: norm std(1.872231) mean(0.251125)
[DEBUG]2020-06-18 13:00:33,889:utils:params info pg1: norm std(10.842709) mean(8.226810)
[DEBUG]2020-06-18 13:00:33,890:utils:grad info pg1: norm std(0.049335) mean(0.011875)
[DEBUG]2020-06-18 13:00:35,080:utils:Train Step 7, bs: 16, loss: 1.05266, lr: 4e-05 final_score: 0.53182, mc_score: 0.11745, time: 114.91363
[DEBUG]2020-06-18 13:00:50,486:utils:params info pg0: norm std(212.533722) mean(76.280647)
[DEBUG]2020-06-18 13:00:50,488:utils:grad info pg0: norm std(0.208455) mean(0.035497)
[DEBUG]2020-06-18 13:00:50,495:utils:params info pg1: norm std(10.842694) mean(8.226801)
[DEBUG]2020-06-18 13:00:50,496:utils:grad info pg1: norm std(0.007367) mean(0.001841)
[DEBUG]2020-06-18 13:00:51,679:utils:Train Step 8, bs: 16, loss: 1.01540, lr: 4e-05 final_score: 0.55123, mc_score: 0.14864, time: 131.51252
[DEBUG]2020-06-18 13:01:07,078:utils:params info pg0: norm std(212.533722) mean(76.280663)
[DEBUG]2020-06-18 13:01:07,080:utils:grad info pg0: norm std(0.995567) mean(0.118626)
[DEBUG]2020-06-18 13:01:07,087:utils:params info pg1: norm std(10.842683) mean(8.226792)
[DEBUG]2020-06-18 13:01:07,088:utils:grad info pg1: norm std(0.021725) mean(0.004558)
[DEBUG]2020-06-18 13:01:08,308:utils:Train Step 9, bs: 16, loss: 1.01310, lr: 4e-05 final_score: 0.54431, mc_score: 0.11455, time: 148.14178
[DEBUG]2020-06-18 13:01:23,536:utils:params info pg0: norm std(212.533737) mean(76.280678)
[DEBUG]2020-06-18 13:01:23,538:utils:grad info pg0: norm std(0.985859) mean(0.130040)
[DEBUG]2020-06-18 13:01:23,545:utils:params info pg1: norm std(10.842670) mean(8.226786)
[DEBUG]2020-06-18 13:01:23,546:utils:grad info pg1: norm std(0.019594) mean(0.004934)
[DEBUG]2020-06-18 13:01:24,716:utils:Train Step 10, bs: 16, loss: 0.99814, lr: 4e-05 final_score: 0.54997, mc_score: 0.12516, time: 164.54980
[DEBUG]2020-06-18 13:01:40,164:utils:params info pg0: norm std(212.533737) mean(76.280685)
[DEBUG]2020-06-18 13:01:40,166:utils:grad info pg0: norm std(0.892248) mean(0.126503)
[DEBUG]2020-06-18 13:01:40,173:utils:params info pg1: norm std(10.842652) mean(8.226779)
[DEBUG]2020-06-18 13:01:40,174:utils:grad info pg1: norm std(0.026747) mean(0.006088)
[DEBUG]2020-06-18 13:01:41,354:utils:Train Step 11, bs: 16, loss: 0.98994, lr: 4e-05 final_score: 0.55849, mc_score: 0.13371, time: 181.18776
[DEBUG]2020-06-18 13:01:56,521:utils:params info pg0: norm std(212.533737) mean(76.280685)
[DEBUG]2020-06-18 13:01:56,523:utils:grad info pg0: norm std(1.472403) mean(0.185932)
[DEBUG]2020-06-18 13:01:56,530:utils:params info pg1: norm std(10.842631) mean(8.226772)
[DEBUG]2020-06-18 13:01:56,531:utils:grad info pg1: norm std(0.036370) mean(0.007857)
[DEBUG]2020-06-18 13:01:57,731:utils:Train Step 12, bs: 16, loss: 1.00072, lr: 4e-05 final_score: 0.54307, mc_score: 0.11199, time: 197.56456
[DEBUG]2020-06-18 13:02:13,156:utils:params info pg0: norm std(212.533737) mean(76.280693)
[DEBUG]2020-06-18 13:02:13,158:utils:grad info pg0: norm std(1.414801) mean(0.165110)
[DEBUG]2020-06-18 13:02:13,165:utils:params info pg1: norm std(10.842613) mean(8.226766)
[DEBUG]2020-06-18 13:02:13,166:utils:grad info pg1: norm std(0.031508) mean(0.006507)
[DEBUG]2020-06-18 13:02:14,373:utils:Train Step 13, bs: 16, loss: 1.00205, lr: 4e-05 final_score: 0.53176, mc_score: 0.08609, time: 214.20676
[DEBUG]2020-06-18 13:02:29,732:utils:params info pg0: norm std(212.533768) mean(76.280701)
[DEBUG]2020-06-18 13:02:29,734:utils:grad info pg0: norm std(1.336432) mean(0.158541)
[DEBUG]2020-06-18 13:02:29,742:utils:params info pg1: norm std(10.842599) mean(8.226758)
[DEBUG]2020-06-18 13:02:29,743:utils:grad info pg1: norm std(0.028287) mean(0.005830)
[DEBUG]2020-06-18 13:02:31,018:utils:Train Step 14, bs: 16, loss: 0.99254, lr: 4e-05 final_score: 0.53437, mc_score: 0.08405, time: 230.85220
[DEBUG]2020-06-18 13:02:46,485:utils:params info pg0: norm std(212.533768) mean(76.280708)
[DEBUG]2020-06-18 13:02:46,487:utils:grad info pg0: norm std(0.814592) mean(0.110430)
[DEBUG]2020-06-18 13:02:46,494:utils:params info pg1: norm std(10.842587) mean(8.226753)
[DEBUG]2020-06-18 13:02:46,495:utils:grad info pg1: norm std(0.021436) mean(0.004854)
[DEBUG]2020-06-18 13:02:47,744:utils:Train Step 15, bs: 16, loss: 0.99344, lr: 4e-05 final_score: 0.52904, mc_score: 0.08544, time: 247.57823
```
RuntimeError: CUDA out of memory. Tried to allocate 14.00 MiB (GPU 0; 15.90 GiB
total capacity; 14.92 GiB already allocated; 13.88 MiB free; 15.10 GiB reserved
in total by PyTorch)

TPU xla

```
step: 0, loss: 1.054688
step: 0, loss: 1.257812
step: 0, loss: 1.445312
step: 0, loss: 0.910156
step: 0, loss: 1.109375
step: 0, loss: 1.062500
step: 0, loss: 1.101562
step: 0, loss: 1.101562
```

```
Train Step 147, bs: 16, loss: 0.68361, lr: 4e-05 final_score: 0.90438, mc_score: 0.64039, time: 467.80383
Train Step 147, bs: 16, loss: 0.67254, lr: 4e-05 final_score: 0.91099, mc_score: 0.65733, time: 468.97236
Train Step 147, bs: 16, loss: 0.67780, lr: 4e-05 final_score: 0.90746, mc_score: 0.64567, time: 468.05209
Train Step 147, bs: 16, loss: 0.68832, lr: 4e-05 final_score: 0.89785, mc_score: 0.62363, time: 471.52067
Train Step 147, bs: 16, loss: 0.70051, lr: 4e-05 final_score: 0.89038, mc_score: 0.60921, time: 467.14177
Train Step 147, bs: 16, loss: 0.66625, lr: 4e-05 final_score: 0.91451, mc_score: 0.66745, time: 471.24668
Train Step 147, bs: 16, loss: 0.66263, lr: 4e-05 final_score: 0.91621, mc_score: 0.67335, time: 483.16286
Train Step 147, bs: 16, loss: 0.67306, lr: 4e-05 final_score: 0.90880, mc_score: 0.65424, time: 522.60901
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 147, loss: 0.703125
step: 147, loss: 0.511719
step: 147, loss: 0.550781
step: 147, loss: 0.511719
step: 147, loss: 0.574219
step: 147, loss: 0.640625
step: 147, loss: 0.578125
step: 147, loss: 0.699219
Train Step 148, bs: 16, loss: 0.68245, lr: 4e-05 final_score: 0.90520, mc_score: 0.64284, time: 469.83344
Train Step 148, bs: 16, loss: 0.67172, lr: 4e-05 final_score: 0.91140, mc_score: 0.65877, time: 471.00267
Train Step 148, bs: 16, loss: 0.66521, lr: 4e-05 final_score: 0.91516, mc_score: 0.66890, time: 473.27594
Train Step 148, bs: 16, loss: 0.67713, lr: 4e-05 final_score: 0.90795, mc_score: 0.64641, time: 470.08089
Train Step 148, bs: 16, loss: 0.66290, lr: 4e-05 final_score: 0.91585, mc_score: 0.67386, time: 485.19144
Train Step 148, bs: 16, loss: 0.70010, lr: 4e-05 final_score: 0.89046, mc_score: 0.61013, time: 469.17217
Train Step 148, bs: 16, loss: 0.68755, lr: 4e-05 final_score: 0.89837, mc_score: 0.62533, time: 473.55186
Train Step 148, bs: 16, loss: 0.67323, lr: 4e-05 final_score: 0.90858, mc_score: 0.65489, time: 524.65650
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 148, loss: 0.597656
step: 148, loss: 0.683594
step: 148, loss: 0.644531
step: 148, loss: 0.617188
step: 148, loss: 0.628906
step: 148, loss: 0.589844
step: 148, loss: 0.664062
step: 148, loss: 0.605469
Train Step 149, bs: 16, loss: 0.68695, lr: 4e-05 final_score: 0.89885, mc_score: 0.62701, time: 475.56582
Train Step 149, bs: 16, loss: 0.66489, lr: 4e-05 final_score: 0.91530, mc_score: 0.66777, time: 475.29355
Train Step 149, bs: 16, loss: 0.69963, lr: 4e-05 final_score: 0.89077, mc_score: 0.61107, time: 471.19232
Train Step 149, bs: 16, loss: 0.66241, lr: 4e-05 final_score: 0.91616, mc_score: 0.67438, time: 487.23053
Train Step 149, bs: 16, loss: 0.67317, lr: 4e-05 final_score: 0.90856, mc_score: 0.65388, time: 526.68332
Train Step 149, bs: 16, loss: 0.67665, lr: 4e-05 final_score: 0.90821, mc_score: 0.64628, time: 472.12707
Train Step 149, bs: 16, loss: 0.68219, lr: 4e-05 final_score: 0.90534, mc_score: 0.64276, time: 471.89951
Train Step 149, bs: 16, loss: 0.67180, lr: 4e-05 final_score: 0.91137, mc_score: 0.65938, time: 473.06898
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 149, loss: 0.558594
step: 149, loss: 0.550781
step: 149, loss: 0.585938
step: 149, loss: 0.511719
step: 149, loss: 0.656250
step: 149, loss: 0.500000
step: 149, loss: 0.636719
step: 149, loss: 0.664062
Train Step 150, bs: 16, loss: 0.68628, lr: 4e-05 final_score: 0.89930, mc_score: 0.62861, time: 477.63172
Train Step 150, bs: 16, loss: 0.66172, lr: 4e-05 final_score: 0.91658, mc_score: 0.67495, time: 489.27289
Train Step 150, bs: 16, loss: 0.66413, lr: 4e-05 final_score: 0.91609, mc_score: 0.66997, time: 477.35929
Train Step 150, bs: 16, loss: 0.67202, lr: 4e-05 final_score: 0.90935, mc_score: 0.65620, time: 528.72887
Train Step 150, bs: 16, loss: 0.68189, lr: 4e-05 final_score: 0.90554, mc_score: 0.64347, time: 473.94420
Train Step 150, bs: 16, loss: 0.67174, lr: 4e-05 final_score: 0.91142, mc_score: 0.66003, time: 475.12498
Train Step 150, bs: 16, loss: 0.67651, lr: 4e-05 final_score: 0.90820, mc_score: 0.64701, time: 474.21225
Train Step 150, bs: 16, loss: 0.69837, lr: 4e-05 final_score: 0.89167, mc_score: 0.61366, time: 473.30248
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 150, loss: 0.500000
step: 150, loss: 0.523438
step: 150, loss: 0.628906
step: 150, loss: 0.488281
step: 150, loss: 0.574219
step: 150, loss: 0.628906
step: 150, loss: 0.664062
step: 150, loss: 0.601562
Train Step 151, bs: 16, loss: 0.67076, lr: 4e-05 final_score: 0.91201, mc_score: 0.66143, time: 477.16060
Train Step 151, bs: 16, loss: 0.66065, lr: 4e-05 final_score: 0.91734, mc_score: 0.67709, time: 491.34803
Train Step 151, bs: 16, loss: 0.67173, lr: 4e-05 final_score: 0.90968, mc_score: 0.65768, time: 530.80837
Train Step 151, bs: 16, loss: 0.66353, lr: 4e-05 final_score: 0.91637, mc_score: 0.67130, time: 479.44742
Train Step 151, bs: 16, loss: 0.68590, lr: 4e-05 final_score: 0.89947, mc_score: 0.62944, time: 479.72094
Train Step 151, bs: 16, loss: 0.68061, lr: 4e-05 final_score: 0.90638, mc_score: 0.64588, time: 476.00348
Train Step 151, bs: 16, loss: 0.67601, lr: 4e-05 final_score: 0.90854, mc_score: 0.64763, time: 476.33032
Train Step 151, bs: 16, loss: 0.69815, lr: 4e-05 final_score: 0.89181, mc_score: 0.61456, time: 475.41324
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 151, loss: 0.644531
step: 151, loss: 0.667969
step: 151, loss: 0.757812
step: 151, loss: 0.507812
step: 151, loss: 0.589844
step: 151, loss: 0.546875
step: 151, loss: 0.503906
step: 151, loss: 0.589844
Train Step 152, bs: 16, loss: 0.67065, lr: 4e-05 final_score: 0.91044, mc_score: 0.65991, time: 532.90260
Train Step 152, bs: 16, loss: 0.67059, lr: 4e-05 final_score: 0.91205, mc_score: 0.66122, time: 479.26925
Train Step 152, bs: 16, loss: 0.68578, lr: 4e-05 final_score: 0.89957, mc_score: 0.62931, time: 481.81610
Train Step 152, bs: 16, loss: 0.67545, lr: 4e-05 final_score: 0.90878, mc_score: 0.64834, time: 478.34865
Train Step 152, bs: 16, loss: 0.69854, lr: 4e-05 final_score: 0.89145, mc_score: 0.61387, time: 477.43828
Train Step 152, bs: 16, loss: 0.65990, lr: 4e-05 final_score: 0.91769, mc_score: 0.67837, time: 493.46358
Train Step 152, bs: 16, loss: 0.67944, lr: 4e-05 final_score: 0.90712, mc_score: 0.64814, time: 478.11489
Train Step 152, bs: 16, loss: 0.66305, lr: 4e-05 final_score: 0.91664, mc_score: 0.67258, time: 481.55760
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 152, loss: 0.498047
step: 152, loss: 0.726562
step: 152, loss: 0.664062
step: 152, loss: 0.500000
step: 152, loss: 0.781250
step: 152, loss: 0.515625
step: 152, loss: 0.609375
step: 152, loss: 0.589844
Train Step 153, bs: 16, loss: 0.65884, lr: 4e-05 final_score: 0.91837, mc_score: 0.68048, time: 495.48955
Train Step 153, bs: 16, loss: 0.69831, lr: 4e-05 final_score: 0.89163, mc_score: 0.61560, time: 479.47074
Train Step 153, bs: 16, loss: 0.67096, lr: 4e-05 final_score: 0.91176, mc_score: 0.66022, time: 481.30364
Train Step 153, bs: 16, loss: 0.67430, lr: 4e-05 final_score: 0.90957, mc_score: 0.65064, time: 480.38273
Train Step 153, bs: 16, loss: 0.68640, lr: 4e-05 final_score: 0.89903, mc_score: 0.63023, time: 483.85352
Train Step 153, bs: 16, loss: 0.66964, lr: 4e-05 final_score: 0.91113, mc_score: 0.66214, time: 534.94257
Train Step 153, bs: 16, loss: 0.67899, lr: 4e-05 final_score: 0.90726, mc_score: 0.64966, time: 480.15010
Train Step 153, bs: 16, loss: 0.66257, lr: 4e-05 final_score: 0.91695, mc_score: 0.67393, time: 483.59210
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 153, loss: 0.539062
step: 153, loss: 0.632812
step: 153, loss: 0.531250
step: 153, loss: 0.500000
step: 153, loss: 0.542969
step: 153, loss: 0.617188
step: 153, loss: 0.484375
step: 153, loss: 0.523438
Train Step 154, bs: 16, loss: 0.67010, lr: 4e-05 final_score: 0.91236, mc_score: 0.66243, time: 483.34629
Train Step 154, bs: 16, loss: 0.67317, lr: 4e-05 final_score: 0.91040, mc_score: 0.65290, time: 482.42501
Train Step 154, bs: 16, loss: 0.65867, lr: 4e-05 final_score: 0.91833, mc_score: 0.68092, time: 497.53433
Train Step 154, bs: 16, loss: 0.68539, lr: 4e-05 final_score: 0.89963, mc_score: 0.63190, time: 485.89404
Train Step 154, bs: 16, loss: 0.69730, lr: 4e-05 final_score: 0.89235, mc_score: 0.61732, time: 481.51842
Train Step 154, bs: 16, loss: 0.66167, lr: 4e-05 final_score: 0.91760, mc_score: 0.67605, time: 485.63528
Train Step 154, bs: 16, loss: 0.66930, lr: 4e-05 final_score: 0.91123, mc_score: 0.66181, time: 537.02513
Train Step 154, bs: 16, loss: 0.67772, lr: 4e-05 final_score: 0.90805, mc_score: 0.65197, time: 482.23271
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 154, loss: 0.531250
step: 154, loss: 0.632812
step: 154, loss: 0.578125
step: 154, loss: 0.515625
step: 154, loss: 0.628906
step: 154, loss: 0.625000
step: 154, loss: 0.585938
step: 154, loss: 0.648438
Train Step 155, bs: 16, loss: 0.65785, lr: 4e-05 final_score: 0.91878, mc_score: 0.68221, time: 499.60767
Train Step 155, bs: 16, loss: 0.67288, lr: 4e-05 final_score: 0.91052, mc_score: 0.65432, time: 484.49797
Train Step 155, bs: 16, loss: 0.69613, lr: 4e-05 final_score: 0.89317, mc_score: 0.61978, time: 483.59041
Train Step 155, bs: 16, loss: 0.68506, lr: 4e-05 final_score: 0.89972, mc_score: 0.63334, time: 487.96753
Train Step 155, bs: 16, loss: 0.67713, lr: 4e-05 final_score: 0.90834, mc_score: 0.65345, time: 484.28490
Train Step 155, bs: 16, loss: 0.66951, lr: 4e-05 final_score: 0.91282, mc_score: 0.66377, time: 485.46731
Train Step 155, bs: 16, loss: 0.66143, lr: 4e-05 final_score: 0.91764, mc_score: 0.67651, time: 487.74150
Train Step 155, bs: 16, loss: 0.66917, lr: 4e-05 final_score: 0.91131, mc_score: 0.66234, time: 539.10919
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 155, loss: 0.566406
step: 155, loss: 0.789062
step: 155, loss: 0.593750
step: 155, loss: 0.490234
step: 155, loss: 0.707031
step: 155, loss: 0.648438
step: 155, loss: 0.609375
step: 155, loss: 0.625000
Train Step 156, bs: 16, loss: 0.67593, lr: 4e-05 final_score: 0.90922, mc_score: 0.65565, time: 486.32944
Train Step 156, bs: 16, loss: 0.65726, lr: 4e-05 final_score: 0.91915, mc_score: 0.68348, time: 501.68469
Train Step 156, bs: 16, loss: 0.69620, lr: 4e-05 final_score: 0.89333, mc_score: 0.61991, time: 485.66599
Train Step 156, bs: 16, loss: 0.66903, lr: 4e-05 final_score: 0.91135, mc_score: 0.66285, time: 541.13390
Train Step 156, bs: 16, loss: 0.66912, lr: 4e-05 final_score: 0.91306, mc_score: 0.66354, time: 487.53235
Train Step 156, bs: 16, loss: 0.66120, lr: 4e-05 final_score: 0.91772, mc_score: 0.67617, time: 489.80717
Train Step 156, bs: 16, loss: 0.67363, lr: 4e-05 final_score: 0.90982, mc_score: 0.65341, time: 486.62324
Train Step 156, bs: 16, loss: 0.68447, lr: 4e-05 final_score: 0.90015, mc_score: 0.63494, time: 490.09283
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 156, loss: 0.515625
step: 156, loss: 0.523438
step: 156, loss: 0.675781
step: 156, loss: 0.625000
step: 156, loss: 0.613281
step: 156, loss: 0.550781
step: 156, loss: 0.500000
step: 156, loss: 0.648438
Train Step 157, bs: 16, loss: 0.68344, lr: 4e-05 final_score: 0.90087, mc_score: 0.63729, time: 492.11349
Train Step 157, bs: 16, loss: 0.65636, lr: 4e-05 final_score: 0.91970, mc_score: 0.68466, time: 503.75455
Train Step 157, bs: 16, loss: 0.66916, lr: 4e-05 final_score: 0.91306, mc_score: 0.66251, time: 489.56730
Train Step 157, bs: 16, loss: 0.67332, lr: 4e-05 final_score: 0.91007, mc_score: 0.65485, time: 488.64601
Train Step 157, bs: 16, loss: 0.66828, lr: 4e-05 final_score: 0.91180, mc_score: 0.66421, time: 543.20517
Train Step 157, bs: 16, loss: 0.66017, lr: 4e-05 final_score: 0.91849, mc_score: 0.67749, time: 491.84614
Train Step 157, bs: 16, loss: 0.67576, lr: 4e-05 final_score: 0.90921, mc_score: 0.65632, time: 488.42159
Train Step 157, bs: 16, loss: 0.69567, lr: 4e-05 final_score: 0.89358, mc_score: 0.62074, time: 487.76687
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 157, loss: 0.613281
step: 157, loss: 0.490234
step: 157, loss: 0.539062
step: 157, loss: 0.664062
step: 157, loss: 0.535156
step: 157, loss: 0.542969
step: 157, loss: 0.492188
step: 157, loss: 0.648438
Train Step 158, bs: 16, loss: 0.68300, lr: 4e-05 final_score: 0.90110, mc_score: 0.63794, time: 494.16483
Train Step 158, bs: 16, loss: 0.67326, lr: 4e-05 final_score: 0.91015, mc_score: 0.65546, time: 490.69663
Train Step 158, bs: 16, loss: 0.69466, lr: 4e-05 final_score: 0.89427, mc_score: 0.62238, time: 489.78705
Train Step 158, bs: 16, loss: 0.66749, lr: 4e-05 final_score: 0.91227, mc_score: 0.66554, time: 545.25392
Train Step 158, bs: 16, loss: 0.65911, lr: 4e-05 final_score: 0.91914, mc_score: 0.67955, time: 493.90626
Train Step 158, bs: 16, loss: 0.67558, lr: 4e-05 final_score: 0.90917, mc_score: 0.65686, time: 490.47135
Train Step 158, bs: 16, loss: 0.65562, lr: 4e-05 final_score: 0.92018, mc_score: 0.68589, time: 505.84327
Train Step 158, bs: 16, loss: 0.66803, lr: 4e-05 final_score: 0.91381, mc_score: 0.66468, time: 491.65653
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 158, loss: 0.482422
step: 158, loss: 0.539062
step: 158, loss: 0.671875
step: 158, loss: 0.601562
step: 158, loss: 0.640625
step: 158, loss: 0.828125
step: 158, loss: 0.542969
step: 158, loss: 0.695312
Train Step 159, bs: 16, loss: 0.65453, lr: 4e-05 final_score: 0.92083, mc_score: 0.68786, time: 507.86162
Train Step 159, bs: 16, loss: 0.69407, lr: 4e-05 final_score: 0.89462, mc_score: 0.62399, time: 491.84424
Train Step 159, bs: 16, loss: 0.68293, lr: 4e-05 final_score: 0.90092, mc_score: 0.63788, time: 496.22397
Train Step 159, bs: 16, loss: 0.67423, lr: 4e-05 final_score: 0.90945, mc_score: 0.65440, time: 492.75613
Train Step 159, bs: 16, loss: 0.66732, lr: 4e-05 final_score: 0.91231, mc_score: 0.66609, time: 547.31224
Train Step 159, bs: 16, loss: 0.66722, lr: 4e-05 final_score: 0.91439, mc_score: 0.66603, time: 493.67356
Train Step 159, bs: 16, loss: 0.65838, lr: 4e-05 final_score: 0.91951, mc_score: 0.67996, time: 495.96514
Train Step 159, bs: 16, loss: 0.67571, lr: 4e-05 final_score: 0.90936, mc_score: 0.65741, time: 492.52605
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 159, loss: 0.648438
step: 159, loss: 0.546875
step: 159, loss: 0.566406
step: 159, loss: 0.531250
step: 159, loss: 0.574219
step: 159, loss: 0.574219
step: 159, loss: 0.570312
step: 159, loss: 0.703125
Train Step 160, bs: 16, loss: 0.67356, lr: 4e-05 final_score: 0.90999, mc_score: 0.65576, time: 494.80146
Train Step 160, bs: 16, loss: 0.66647, lr: 4e-05 final_score: 0.91280, mc_score: 0.66740, time: 549.35773
Train Step 160, bs: 16, loss: 0.65449, lr: 4e-05 final_score: 0.92078, mc_score: 0.68756, time: 509.91181
Train Step 160, bs: 16, loss: 0.69332, lr: 4e-05 final_score: 0.89505, mc_score: 0.62559, time: 493.89261
Train Step 160, bs: 16, loss: 0.66647, lr: 4e-05 final_score: 0.91497, mc_score: 0.66732, time: 495.72512
Train Step 160, bs: 16, loss: 0.68225, lr: 4e-05 final_score: 0.90139, mc_score: 0.63854, time: 498.27333
Train Step 160, bs: 16, loss: 0.67588, lr: 4e-05 final_score: 0.90928, mc_score: 0.65733, time: 494.56797
Train Step 160, bs: 16, loss: 0.65782, lr: 4e-05 final_score: 0.91981, mc_score: 0.68119, time: 498.01158
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 160, loss: 0.500000
step: 160, loss: 0.535156
step: 160, loss: 0.609375
step: 160, loss: 0.605469
step: 160, loss: 0.582031
step: 160, loss: 0.644531
step: 160, loss: 0.494141
step: 160, loss: 0.550781
Train Step 161, bs: 16, loss: 0.66565, lr: 4e-05 final_score: 0.91551, mc_score: 0.66940, time: 497.75647
Train Step 161, bs: 16, loss: 0.65353, lr: 4e-05 final_score: 0.92138, mc_score: 0.68949, time: 511.94496
Train Step 161, bs: 16, loss: 0.66611, lr: 4e-05 final_score: 0.91292, mc_score: 0.66786, time: 551.39188
Train Step 161, bs: 16, loss: 0.67338, lr: 4e-05 final_score: 0.91001, mc_score: 0.65632, time: 496.83582
Train Step 161, bs: 16, loss: 0.69263, lr: 4e-05 final_score: 0.89535, mc_score: 0.62548, time: 495.92640
Train Step 161, bs: 16, loss: 0.68177, lr: 4e-05 final_score: 0.90169, mc_score: 0.63846, time: 500.30520
Train Step 161, bs: 16, loss: 0.67475, lr: 4e-05 final_score: 0.90996, mc_score: 0.65950, time: 496.60810
Train Step 161, bs: 16, loss: 0.65716, lr: 4e-05 final_score: 0.92016, mc_score: 0.68239, time: 500.05460
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 161, loss: 0.558594
step: 161, loss: 0.683594
step: 161, loss: 0.656250
step: 161, loss: 0.566406
step: 161, loss: 0.550781
step: 161, loss: 0.769531
step: 161, loss: 0.777344
step: 161, loss: 0.683594
Train Step 162, bs: 16, loss: 0.66576, lr: 4e-05 final_score: 0.91540, mc_score: 0.66910, time: 499.78229
Train Step 162, bs: 16, loss: 0.65295, lr: 4e-05 final_score: 0.92207, mc_score: 0.69140, time: 513.96999
Train Step 162, bs: 16, loss: 0.66540, lr: 4e-05 final_score: 0.91345, mc_score: 0.66836, time: 553.41761
Train Step 162, bs: 16, loss: 0.69185, lr: 4e-05 final_score: 0.89603, mc_score: 0.62698, time: 497.95176
Train Step 162, bs: 16, loss: 0.68162, lr: 4e-05 final_score: 0.90183, mc_score: 0.63933, time: 502.33078
Train Step 162, bs: 16, loss: 0.67397, lr: 4e-05 final_score: 0.90958, mc_score: 0.65614, time: 498.89126
Train Step 162, bs: 16, loss: 0.67538, lr: 4e-05 final_score: 0.90941, mc_score: 0.65846, time: 498.64954
Train Step 162, bs: 16, loss: 0.65732, lr: 4e-05 final_score: 0.92002, mc_score: 0.68282, time: 502.09236
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 162, loss: 0.542969
step: 162, loss: 0.632812
step: 162, loss: 0.519531
step: 162, loss: 0.558594
step: 162, loss: 0.523438
step: 162, loss: 0.515625
step: 162, loss: 0.542969
step: 162, loss: 0.531250
Train Step 163, bs: 16, loss: 0.66556, lr: 4e-05 final_score: 0.91552, mc_score: 0.66955, time: 501.84387
Train Step 163, bs: 16, loss: 0.67303, lr: 4e-05 final_score: 0.91041, mc_score: 0.65826, time: 500.92357
Train Step 163, bs: 16, loss: 0.66448, lr: 4e-05 final_score: 0.91411, mc_score: 0.67038, time: 555.47982
Train Step 163, bs: 16, loss: 0.65227, lr: 4e-05 final_score: 0.92251, mc_score: 0.69182, time: 516.03138
Train Step 163, bs: 16, loss: 0.69082, lr: 4e-05 final_score: 0.89700, mc_score: 0.62927, time: 500.01426
Train Step 163, bs: 16, loss: 0.68086, lr: 4e-05 final_score: 0.90243, mc_score: 0.64068, time: 504.39201
Train Step 163, bs: 16, loss: 0.67450, lr: 4e-05 final_score: 0.91005, mc_score: 0.65901, time: 500.69727
Train Step 163, bs: 16, loss: 0.65662, lr: 4e-05 final_score: 0.92046, mc_score: 0.68401, time: 504.13511
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 163, loss: 0.558594
step: 163, loss: 0.578125
step: 163, loss: 0.722656
step: 163, loss: 0.621094
step: 163, loss: 0.476562
step: 163, loss: 0.535156
step: 163, loss: 0.656250
step: 163, loss: 0.519531
Train Step 164, bs: 16, loss: 0.66491, lr: 4e-05 final_score: 0.91598, mc_score: 0.67084, time: 503.89457
Train Step 164, bs: 16, loss: 0.68951, lr: 4e-05 final_score: 0.89790, mc_score: 0.63153, time: 502.06286
Train Step 164, bs: 16, loss: 0.68112, lr: 4e-05 final_score: 0.90236, mc_score: 0.64057, time: 506.44200
Train Step 164, bs: 16, loss: 0.65182, lr: 4e-05 final_score: 0.92284, mc_score: 0.69217, time: 518.08313
Train Step 164, bs: 16, loss: 0.66422, lr: 4e-05 final_score: 0.91440, mc_score: 0.67163, time: 557.53053
Train Step 164, bs: 16, loss: 0.67218, lr: 4e-05 final_score: 0.91095, mc_score: 0.65956, time: 502.97601
Train Step 164, bs: 16, loss: 0.65662, lr: 4e-05 final_score: 0.92059, mc_score: 0.68438, time: 506.17485
Train Step 164, bs: 16, loss: 0.67355, lr: 4e-05 final_score: 0.91071, mc_score: 0.66025, time: 502.75714
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 164, loss: 0.492188
step: 164, loss: 0.574219
step: 164, loss: 0.695312
step: 164, loss: 0.539062
step: 164, loss: 0.656250
step: 164, loss: 0.597656
step: 164, loss: 0.632812
step: 164, loss: 0.542969
Train Step 165, bs: 16, loss: 0.66386, lr: 4e-05 final_score: 0.91683, mc_score: 0.67283, time: 505.93918
Train Step 165, bs: 16, loss: 0.65135, lr: 4e-05 final_score: 0.92317, mc_score: 0.69246, time: 520.13311
Train Step 165, bs: 16, loss: 0.67209, lr: 4e-05 final_score: 0.91101, mc_score: 0.66015, time: 505.03279
Train Step 165, bs: 16, loss: 0.68896, lr: 4e-05 final_score: 0.89846, mc_score: 0.63303, time: 504.12368
Train Step 165, bs: 16, loss: 0.68120, lr: 4e-05 final_score: 0.90247, mc_score: 0.64121, time: 508.50608
Train Step 165, bs: 16, loss: 0.67276, lr: 4e-05 final_score: 0.91130, mc_score: 0.66154, time: 504.79762
Train Step 165, bs: 16, loss: 0.66346, lr: 4e-05 final_score: 0.91499, mc_score: 0.67281, time: 559.62591
Train Step 165, bs: 16, loss: 0.65647, lr: 4e-05 final_score: 0.92065, mc_score: 0.68407, time: 508.27935
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 165, loss: 0.488281
step: 165, loss: 0.558594
step: 165, loss: 0.478516
step: 165, loss: 0.652344
step: 165, loss: 0.546875
step: 165, loss: 0.476562
step: 165, loss: 0.550781
step: 165, loss: 0.490234
Train Step 166, bs: 16, loss: 0.66280, lr: 4e-05 final_score: 0.91770, mc_score: 0.67478, time: 508.02916
Train Step 166, bs: 16, loss: 0.65079, lr: 4e-05 final_score: 0.92374, mc_score: 0.69358, time: 522.21779
Train Step 166, bs: 16, loss: 0.68874, lr: 4e-05 final_score: 0.89854, mc_score: 0.63229, time: 506.20066
Train Step 166, bs: 16, loss: 0.67133, lr: 4e-05 final_score: 0.91159, mc_score: 0.66145, time: 507.11134
Train Step 166, bs: 16, loss: 0.67998, lr: 4e-05 final_score: 0.90347, mc_score: 0.64332, time: 510.58336
Train Step 166, bs: 16, loss: 0.67158, lr: 4e-05 final_score: 0.91203, mc_score: 0.66356, time: 506.88192
Train Step 166, bs: 16, loss: 0.65584, lr: 4e-05 final_score: 0.92114, mc_score: 0.68521, time: 510.32580
Train Step 166, bs: 16, loss: 0.66242, lr: 4e-05 final_score: 0.91573, mc_score: 0.67478, time: 561.69385
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 166, loss: 0.687500
step: 166, loss: 0.482422
step: 166, loss: 0.812500
step: 166, loss: 0.753906
step: 166, loss: 0.486328
step: 166, loss: 0.781250
step: 166, loss: 0.480469
step: 166, loss: 0.585938
Train Step 167, bs: 16, loss: 0.67218, lr: 4e-05 final_score: 0.91098, mc_score: 0.65975, time: 509.16296
Train Step 167, bs: 16, loss: 0.68059, lr: 4e-05 final_score: 0.90297, mc_score: 0.64326, time: 512.63346
Train Step 167, bs: 16, loss: 0.68750, lr: 4e-05 final_score: 0.89948, mc_score: 0.63448, time: 508.25298
Train Step 167, bs: 16, loss: 0.66296, lr: 4e-05 final_score: 0.91529, mc_score: 0.67310, time: 563.71948
Train Step 167, bs: 16, loss: 0.65101, lr: 4e-05 final_score: 0.92353, mc_score: 0.69319, time: 524.27164
Train Step 167, bs: 16, loss: 0.67044, lr: 4e-05 final_score: 0.91285, mc_score: 0.66558, time: 508.92977
Train Step 167, bs: 16, loss: 0.65542, lr: 4e-05 final_score: 0.92145, mc_score: 0.68551, time: 512.37341
Train Step 167, bs: 16, loss: 0.66175, lr: 4e-05 final_score: 0.91843, mc_score: 0.67672, time: 510.10791
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 167, loss: 0.500000
step: 167, loss: 0.468750
step: 167, loss: 0.519531
step: 167, loss: 0.671875
step: 167, loss: 0.781250
step: 167, loss: 0.617188
step: 167, loss: 0.593750
step: 167, loss: 0.597656
Train Step 168, bs: 16, loss: 0.66942, lr: 4e-05 final_score: 0.91370, mc_score: 0.66760, time: 510.96006
Train Step 168, bs: 16, loss: 0.64993, lr: 4e-05 final_score: 0.92426, mc_score: 0.69501, time: 526.31757
Train Step 168, bs: 16, loss: 0.66211, lr: 4e-05 final_score: 0.91611, mc_score: 0.67506, time: 565.76446
Train Step 168, bs: 16, loss: 0.68021, lr: 4e-05 final_score: 0.90321, mc_score: 0.64313, time: 514.68952
Train Step 168, bs: 16, loss: 0.66134, lr: 4e-05 final_score: 0.91873, mc_score: 0.67787, time: 512.15969
Train Step 168, bs: 16, loss: 0.65508, lr: 4e-05 final_score: 0.92165, mc_score: 0.68663, time: 514.43606
Train Step 168, bs: 16, loss: 0.67283, lr: 4e-05 final_score: 0.91055, mc_score: 0.66032, time: 511.25289
Train Step 168, bs: 16, loss: 0.68741, lr: 4e-05 final_score: 0.89947, mc_score: 0.63516, time: 510.34331
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 168, loss: 0.621094
step: 168, loss: 0.726562
step: 168, loss: 0.539062
step: 168, loss: 0.574219
step: 168, loss: 0.703125
step: 168, loss: 0.484375
step: 168, loss: 0.554688
step: 168, loss: 0.539062
Train Step 169, bs: 16, loss: 0.66110, lr: 4e-05 final_score: 0.91886, mc_score: 0.67899, time: 514.19919
Train Step 169, bs: 16, loss: 0.66138, lr: 4e-05 final_score: 0.91652, mc_score: 0.67623, time: 567.83570
Train Step 169, bs: 16, loss: 0.65038, lr: 4e-05 final_score: 0.92400, mc_score: 0.69457, time: 528.39061
Train Step 169, bs: 16, loss: 0.67958, lr: 4e-05 final_score: 0.90377, mc_score: 0.64446, time: 516.75047
Train Step 169, bs: 16, loss: 0.66962, lr: 4e-05 final_score: 0.91359, mc_score: 0.66735, time: 513.05773
Train Step 169, bs: 16, loss: 0.67171, lr: 4e-05 final_score: 0.91135, mc_score: 0.66232, time: 513.30496
Train Step 169, bs: 16, loss: 0.65448, lr: 4e-05 final_score: 0.92200, mc_score: 0.68706, time: 516.50155
Train Step 169, bs: 16, loss: 0.68653, lr: 4e-05 final_score: 0.90016, mc_score: 0.63730, time: 512.41213
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 169, loss: 0.535156
step: 169, loss: 0.503906
step: 169, loss: 0.562500
step: 169, loss: 0.531250
step: 169, loss: 0.527344
step: 169, loss: 0.578125
step: 169, loss: 0.621094
step: 169, loss: 0.570312
Train Step 170, bs: 16, loss: 0.64970, lr: 4e-05 final_score: 0.92442, mc_score: 0.69560, time: 530.45606
Train Step 170, bs: 16, loss: 0.67073, lr: 4e-05 final_score: 0.91202, mc_score: 0.66431, time: 515.34674
Train Step 170, bs: 16, loss: 0.66034, lr: 4e-05 final_score: 0.91940, mc_score: 0.68018, time: 516.26969
Train Step 170, bs: 16, loss: 0.68589, lr: 4e-05 final_score: 0.90064, mc_score: 0.63871, time: 514.43981
Train Step 170, bs: 16, loss: 0.66934, lr: 4e-05 final_score: 0.91383, mc_score: 0.66787, time: 515.12459
Train Step 170, bs: 16, loss: 0.65399, lr: 4e-05 final_score: 0.92226, mc_score: 0.68743, time: 518.56793
Train Step 170, bs: 16, loss: 0.67889, lr: 4e-05 final_score: 0.90415, mc_score: 0.64437, time: 518.86375
Train Step 170, bs: 16, loss: 0.66059, lr: 4e-05 final_score: 0.91698, mc_score: 0.67815, time: 569.95196
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 170, loss: 0.507812
step: 170, loss: 0.482422
step: 170, loss: 0.648438
step: 170, loss: 0.691406
step: 170, loss: 0.519531
step: 170, loss: 0.632812
step: 170, loss: 0.511719
step: 170, loss: 0.828125
Train Step 171, bs: 16, loss: 0.68492, lr: 4e-05 final_score: 0.90141, mc_score: 0.64081, time: 516.50656
Train Step 171, bs: 16, loss: 0.65972, lr: 4e-05 final_score: 0.91754, mc_score: 0.67933, time: 571.97307
Train Step 171, bs: 16, loss: 0.67863, lr: 4e-05 final_score: 0.90432, mc_score: 0.64492, time: 520.88563
Train Step 171, bs: 16, loss: 0.66027, lr: 4e-05 final_score: 0.91950, mc_score: 0.67989, time: 518.33831
Train Step 171, bs: 16, loss: 0.64872, lr: 4e-05 final_score: 0.92509, mc_score: 0.69737, time: 532.52885
Train Step 171, bs: 16, loss: 0.65421, lr: 4e-05 final_score: 0.92192, mc_score: 0.68784, time: 520.61254
Train Step 171, bs: 16, loss: 0.67165, lr: 4e-05 final_score: 0.91137, mc_score: 0.66406, time: 517.42210
Train Step 171, bs: 16, loss: 0.66839, lr: 4e-05 final_score: 0.91440, mc_score: 0.66982, time: 517.17004
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 171, loss: 0.667969
step: 171, loss: 0.578125
step: 171, loss: 0.593750
step: 171, loss: 0.484375
step: 171, loss: 0.523438
step: 171, loss: 0.648438
step: 171, loss: 0.574219
step: 171, loss: 0.523438
Train Step 172, bs: 16, loss: 0.66839, lr: 4e-05 final_score: 0.91444, mc_score: 0.67037, time: 519.20730
Train Step 172, bs: 16, loss: 0.68439, lr: 4e-05 final_score: 0.90175, mc_score: 0.64220, time: 518.54541
Train Step 172, bs: 16, loss: 0.64831, lr: 4e-05 final_score: 0.92528, mc_score: 0.69838, time: 534.56505
Train Step 172, bs: 16, loss: 0.65322, lr: 4e-05 final_score: 0.92245, mc_score: 0.68959, time: 522.65132
Train Step 172, bs: 16, loss: 0.67845, lr: 4e-05 final_score: 0.90438, mc_score: 0.64543, time: 523.00156
Train Step 172, bs: 16, loss: 0.67108, lr: 4e-05 final_score: 0.91163, mc_score: 0.66529, time: 519.53342
Train Step 172, bs: 16, loss: 0.65893, lr: 4e-05 final_score: 0.91801, mc_score: 0.68047, time: 574.08953
Train Step 172, bs: 16, loss: 0.65947, lr: 4e-05 final_score: 0.92009, mc_score: 0.68105, time: 520.45516
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 172, loss: 0.550781
step: 172, loss: 0.679688
step: 172, loss: 0.503906
step: 172, loss: 0.500000
step: 172, loss: 0.462891
step: 172, loss: 0.519531
step: 172, loss: 0.511719
step: 172, loss: 0.496094
Train Step 173, bs: 16, loss: 0.66771, lr: 4e-05 final_score: 0.91493, mc_score: 0.67151, time: 521.31488
Train Step 173, bs: 16, loss: 0.64849, lr: 4e-05 final_score: 0.92512, mc_score: 0.69867, time: 536.67149
Train Step 173, bs: 16, loss: 0.68335, lr: 4e-05 final_score: 0.90253, mc_score: 0.64359, time: 520.65306
Train Step 173, bs: 16, loss: 0.65233, lr: 4e-05 final_score: 0.92302, mc_score: 0.69135, time: 524.75983
Train Step 173, bs: 16, loss: 0.65834, lr: 4e-05 final_score: 0.92080, mc_score: 0.68291, time: 522.50131
Train Step 173, bs: 16, loss: 0.67020, lr: 4e-05 final_score: 0.91210, mc_score: 0.66651, time: 521.58158
Train Step 173, bs: 16, loss: 0.65799, lr: 4e-05 final_score: 0.91851, mc_score: 0.68231, time: 576.13928
Train Step 173, bs: 16, loss: 0.67749, lr: 4e-05 final_score: 0.90519, mc_score: 0.64692, time: 525.05170
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 173, loss: 0.585938
step: 173, loss: 0.734375
step: 173, loss: 0.722656
step: 173, loss: 0.898438
step: 173, loss: 0.531250
step: 173, loss: 0.667969
step: 173, loss: 0.605469
step: 173, loss: 0.613281
Train Step 174, bs: 16, loss: 0.66809, lr: 4e-05 final_score: 0.91459, mc_score: 0.67194, time: 523.35971
Train Step 174, bs: 16, loss: 0.68247, lr: 4e-05 final_score: 0.90314, mc_score: 0.64563, time: 522.69748
Train Step 174, bs: 16, loss: 0.64813, lr: 4e-05 final_score: 0.92546, mc_score: 0.69971, time: 538.71724
Train Step 174, bs: 16, loss: 0.67743, lr: 4e-05 final_score: 0.90516, mc_score: 0.64837, time: 527.07720
Train Step 174, bs: 16, loss: 0.65769, lr: 4e-05 final_score: 0.91875, mc_score: 0.68339, time: 578.16489
Train Step 174, bs: 16, loss: 0.65871, lr: 4e-05 final_score: 0.92059, mc_score: 0.68328, time: 524.53086
Train Step 174, bs: 16, loss: 0.67152, lr: 4e-05 final_score: 0.91121, mc_score: 0.66555, time: 523.61019
Train Step 174, bs: 16, loss: 0.65211, lr: 4e-05 final_score: 0.92308, mc_score: 0.69089, time: 526.80836
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 174, loss: 0.500000
step: 174, loss: 0.683594
step: 174, loss: 0.589844
step: 174, loss: 0.609375
step: 174, loss: 0.664062
step: 174, loss: 0.535156
step: 174, loss: 0.511719
step: 174, loss: 0.503906
Train Step 175, bs: 16, loss: 0.66818, lr: 4e-05 final_score: 0.91445, mc_score: 0.67162, time: 525.36850
Train Step 175, bs: 16, loss: 0.67704, lr: 4e-05 final_score: 0.90544, mc_score: 0.64971, time: 529.08495
Train Step 175, bs: 16, loss: 0.67147, lr: 4e-05 final_score: 0.91134, mc_score: 0.66671, time: 525.61641
Train Step 175, bs: 16, loss: 0.68143, lr: 4e-05 final_score: 0.90386, mc_score: 0.64764, time: 524.70484
Train Step 175, bs: 16, loss: 0.64780, lr: 4e-05 final_score: 0.92570, mc_score: 0.70073, time: 540.72939
Train Step 175, bs: 16, loss: 0.65131, lr: 4e-05 final_score: 0.92368, mc_score: 0.69199, time: 528.83689
Train Step 175, bs: 16, loss: 0.65681, lr: 4e-05 final_score: 0.91939, mc_score: 0.68524, time: 580.20686
Train Step 175, bs: 16, loss: 0.65800, lr: 4e-05 final_score: 0.92113, mc_score: 0.68433, time: 526.57767
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 175, loss: 0.480469
step: 175, loss: 0.500000
step: 175, loss: 0.707031
step: 175, loss: 0.687500
step: 175, loss: 0.789062
step: 175, loss: 0.695312
step: 175, loss: 0.582031
step: 175, loss: 0.710938
Train Step 176, bs: 16, loss: 0.65828, lr: 4e-05 final_score: 0.92108, mc_score: 0.68471, time: 528.61693
Train Step 176, bs: 16, loss: 0.64696, lr: 4e-05 final_score: 0.92632, mc_score: 0.70242, time: 542.80450
Train Step 176, bs: 16, loss: 0.66711, lr: 4e-05 final_score: 0.91522, mc_score: 0.67352, time: 527.44927
Train Step 176, bs: 16, loss: 0.67161, lr: 4e-05 final_score: 0.91122, mc_score: 0.66649, time: 527.69603
Train Step 176, bs: 16, loss: 0.67710, lr: 4e-05 final_score: 0.90535, mc_score: 0.64955, time: 531.16468
Train Step 176, bs: 16, loss: 0.65756, lr: 4e-05 final_score: 0.91884, mc_score: 0.68487, time: 582.25458
Train Step 176, bs: 16, loss: 0.65164, lr: 4e-05 final_score: 0.92336, mc_score: 0.69088, time: 530.93724
Train Step 176, bs: 16, loss: 0.68086, lr: 4e-05 final_score: 0.90433, mc_score: 0.64895, time: 526.83183
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 176, loss: 0.527344
step: 176, loss: 0.601562
step: 176, loss: 0.613281
step: 176, loss: 0.531250
step: 176, loss: 0.515625
step: 176, loss: 0.494141
step: 176, loss: 0.671875
step: 176, loss: 0.632812
Train Step 177, bs: 16, loss: 0.64629, lr: 4e-05 final_score: 0.92674, mc_score: 0.70410, time: 544.85623
Train Step 177, bs: 16, loss: 0.67668, lr: 4e-05 final_score: 0.90566, mc_score: 0.65021, time: 533.21580
Train Step 177, bs: 16, loss: 0.67128, lr: 4e-05 final_score: 0.91136, mc_score: 0.66765, time: 529.74791
Train Step 177, bs: 16, loss: 0.65756, lr: 4e-05 final_score: 0.92161, mc_score: 0.68651, time: 530.69352
Train Step 177, bs: 16, loss: 0.65664, lr: 4e-05 final_score: 0.91953, mc_score: 0.68665, time: 584.33094
Train Step 177, bs: 16, loss: 0.66626, lr: 4e-05 final_score: 0.91582, mc_score: 0.67470, time: 529.54910
Train Step 177, bs: 16, loss: 0.68059, lr: 4e-05 final_score: 0.90449, mc_score: 0.64953, time: 528.89545
Train Step 177, bs: 16, loss: 0.65176, lr: 4e-05 final_score: 0.92335, mc_score: 0.69116, time: 533.00982
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 177, loss: 0.816406
step: 177, loss: 0.703125
step: 177, loss: 0.828125
step: 177, loss: 0.527344
step: 177, loss: 0.796875
step: 177, loss: 0.554688
step: 177, loss: 0.914062
step: 177, loss: 0.722656
Train Step 178, bs: 16, loss: 0.67584, lr: 4e-05 final_score: 0.90631, mc_score: 0.65143, time: 535.31361
Train Step 178, bs: 16, loss: 0.67989, lr: 4e-05 final_score: 0.90501, mc_score: 0.65082, time: 530.93500
Train Step 178, bs: 16, loss: 0.64661, lr: 4e-05 final_score: 0.92646, mc_score: 0.70430, time: 546.95461
Train Step 178, bs: 16, loss: 0.65852, lr: 4e-05 final_score: 0.92087, mc_score: 0.68475, time: 532.76707
Train Step 178, bs: 16, loss: 0.65216, lr: 4e-05 final_score: 0.92304, mc_score: 0.69017, time: 535.04270
Train Step 178, bs: 16, loss: 0.66710, lr: 4e-05 final_score: 0.91514, mc_score: 0.67446, time: 531.61936
Train Step 178, bs: 16, loss: 0.67264, lr: 4e-05 final_score: 0.91049, mc_score: 0.66666, time: 531.88242
Train Step 178, bs: 16, loss: 0.65742, lr: 4e-05 final_score: 0.91884, mc_score: 0.68489, time: 586.43914
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 178, loss: 0.601562
step: 178, loss: 0.546875
step: 178, loss: 0.546875
step: 178, loss: 0.482422
step: 178, loss: 0.589844
step: 178, loss: 0.484375
step: 178, loss: 0.746094
step: 178, loss: 0.687500
Train Step 179, bs: 16, loss: 0.65711, lr: 4e-05 final_score: 0.91904, mc_score: 0.68594, time: 588.45547
Train Step 179, bs: 16, loss: 0.65122, lr: 4e-05 final_score: 0.92367, mc_score: 0.69191, time: 537.10369
Train Step 179, bs: 16, loss: 0.66643, lr: 4e-05 final_score: 0.91571, mc_score: 0.67558, time: 533.66068
Train Step 179, bs: 16, loss: 0.67878, lr: 4e-05 final_score: 0.90566, mc_score: 0.65277, time: 532.99953
Train Step 179, bs: 16, loss: 0.65901, lr: 4e-05 final_score: 0.92051, mc_score: 0.68364, time: 534.83226
Train Step 179, bs: 16, loss: 0.67218, lr: 4e-05 final_score: 0.91082, mc_score: 0.66713, time: 533.91148
Train Step 179, bs: 16, loss: 0.67512, lr: 4e-05 final_score: 0.90682, mc_score: 0.65265, time: 537.37832
Train Step 179, bs: 16, loss: 0.64683, lr: 4e-05 final_score: 0.92627, mc_score: 0.70312, time: 549.02520
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 179, loss: 0.730469
step: 179, loss: 0.613281
step: 179, loss: 0.722656
step: 179, loss: 0.792969
step: 179, loss: 0.656250
step: 179, loss: 0.539062
step: 179, loss: 0.667969
step: 179, loss: 0.722656
Train Step 180, bs: 16, loss: 0.66679, lr: 4e-05 final_score: 0.91549, mc_score: 0.67527, time: 535.69893
Train Step 180, bs: 16, loss: 0.65162, lr: 4e-05 final_score: 0.92338, mc_score: 0.69227, time: 539.14138
Train Step 180, bs: 16, loss: 0.67866, lr: 4e-05 final_score: 0.90580, mc_score: 0.65331, time: 535.03673
Train Step 180, bs: 16, loss: 0.64665, lr: 4e-05 final_score: 0.92622, mc_score: 0.70333, time: 551.05617
Train Step 180, bs: 16, loss: 0.67577, lr: 4e-05 final_score: 0.90646, mc_score: 0.65249, time: 539.41563
Train Step 180, bs: 16, loss: 0.65936, lr: 4e-05 final_score: 0.92012, mc_score: 0.68333, time: 536.86911
Train Step 180, bs: 16, loss: 0.67144, lr: 4e-05 final_score: 0.91137, mc_score: 0.66829, time: 535.94786
Train Step 180, bs: 16, loss: 0.65717, lr: 4e-05 final_score: 0.91910, mc_score: 0.68703, time: 590.50484
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 180, loss: 0.558594
step: 180, loss: 0.632812
step: 180, loss: 0.500000
step: 180, loss: 0.507812
step: 180, loss: 0.535156
step: 180, loss: 0.490234
step: 180, loss: 0.535156
step: 180, loss: 0.593750
Train Step 181, bs: 16, loss: 0.66619, lr: 4e-05 final_score: 0.91577, mc_score: 0.67634, time: 537.71020
Train Step 181, bs: 16, loss: 0.64657, lr: 4e-05 final_score: 0.92625, mc_score: 0.70359, time: 553.06714
Train Step 181, bs: 16, loss: 0.67054, lr: 4e-05 final_score: 0.91198, mc_score: 0.66945, time: 537.95824
Train Step 181, bs: 16, loss: 0.67786, lr: 4e-05 final_score: 0.90647, mc_score: 0.65390, time: 537.04839
Train Step 181, bs: 16, loss: 0.65078, lr: 4e-05 final_score: 0.92400, mc_score: 0.69399, time: 541.15379
Train Step 181, bs: 16, loss: 0.65650, lr: 4e-05 final_score: 0.91956, mc_score: 0.68876, time: 592.51723
Train Step 181, bs: 16, loss: 0.65843, lr: 4e-05 final_score: 0.92078, mc_score: 0.68508, time: 538.92441
Train Step 181, bs: 16, loss: 0.67532, lr: 4e-05 final_score: 0.90668, mc_score: 0.65368, time: 541.47420
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 181, loss: 0.710938
step: 181, loss: 0.867188
step: 181, loss: 0.609375
step: 181, loss: 0.558594
step: 181, loss: 0.605469
step: 181, loss: 0.632812
step: 181, loss: 0.500000
step: 181, loss: 0.625000
Train Step 182, bs: 16, loss: 0.64692, lr: 4e-05 final_score: 0.92593, mc_score: 0.70251, time: 555.11650
Train Step 182, bs: 16, loss: 0.65197, lr: 4e-05 final_score: 0.92310, mc_score: 0.69156, time: 543.20306
Train Step 182, bs: 16, loss: 0.67721, lr: 4e-05 final_score: 0.90692, mc_score: 0.65513, time: 539.09821
Train Step 182, bs: 16, loss: 0.67020, lr: 4e-05 final_score: 0.91227, mc_score: 0.66993, time: 540.00902
Train Step 182, bs: 16, loss: 0.66585, lr: 4e-05 final_score: 0.91591, mc_score: 0.67603, time: 539.78269
Train Step 182, bs: 16, loss: 0.65756, lr: 4e-05 final_score: 0.92140, mc_score: 0.68682, time: 540.95283
Train Step 182, bs: 16, loss: 0.67508, lr: 4e-05 final_score: 0.90680, mc_score: 0.65434, time: 543.49995
Train Step 182, bs: 16, loss: 0.65633, lr: 4e-05 final_score: 0.91967, mc_score: 0.68977, time: 594.58772
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 182, loss: 0.640625
step: 182, loss: 0.593750
step: 182, loss: 0.496094
step: 182, loss: 0.492188
step: 182, loss: 0.625000
step: 182, loss: 0.542969
step: 182, loss: 0.656250
step: 182, loss: 0.667969
Train Step 183, bs: 16, loss: 0.66572, lr: 4e-05 final_score: 0.91601, mc_score: 0.67715, time: 541.81092
Train Step 183, bs: 16, loss: 0.67692, lr: 4e-05 final_score: 0.90709, mc_score: 0.65504, time: 541.14853
Train Step 183, bs: 16, loss: 0.64663, lr: 4e-05 final_score: 0.92612, mc_score: 0.70276, time: 557.16820
Train Step 183, bs: 16, loss: 0.67436, lr: 4e-05 final_score: 0.90738, mc_score: 0.65621, time: 545.52802
Train Step 183, bs: 16, loss: 0.65199, lr: 4e-05 final_score: 0.92309, mc_score: 0.69190, time: 545.25439
Train Step 183, bs: 16, loss: 0.65668, lr: 4e-05 final_score: 0.92196, mc_score: 0.68850, time: 542.98025
Train Step 183, bs: 16, loss: 0.66923, lr: 4e-05 final_score: 0.91289, mc_score: 0.67173, time: 542.05915
Train Step 183, bs: 16, loss: 0.65639, lr: 4e-05 final_score: 0.91972, mc_score: 0.68874, time: 596.61573
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 183, loss: 0.734375
step: 183, loss: 0.515625
step: 183, loss: 0.562500
step: 183, loss: 0.625000
step: 183, loss: 0.582031
step: 183, loss: 0.609375
step: 183, loss: 0.578125
step: 183, loss: 0.507812
Train Step 184, bs: 16, loss: 0.65591, lr: 4e-05 final_score: 0.92237, mc_score: 0.69018, time: 545.01986
Train Step 184, bs: 16, loss: 0.67641, lr: 4e-05 final_score: 0.90730, mc_score: 0.65554, time: 543.18870
Train Step 184, bs: 16, loss: 0.65613, lr: 4e-05 final_score: 0.91985, mc_score: 0.68908, time: 598.65528
Train Step 184, bs: 16, loss: 0.65159, lr: 4e-05 final_score: 0.92336, mc_score: 0.69227, time: 547.29422
Train Step 184, bs: 16, loss: 0.66899, lr: 4e-05 final_score: 0.91297, mc_score: 0.67217, time: 544.09916
Train Step 184, bs: 16, loss: 0.67375, lr: 4e-05 final_score: 0.90786, mc_score: 0.65729, time: 547.56775
Train Step 184, bs: 16, loss: 0.66609, lr: 4e-05 final_score: 0.91568, mc_score: 0.67697, time: 543.85320
Train Step 184, bs: 16, loss: 0.64588, lr: 4e-05 final_score: 0.92659, mc_score: 0.70438, time: 559.21330
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 184, loss: 0.585938
step: 184, loss: 0.609375
step: 184, loss: 0.703125
step: 184, loss: 0.507812
step: 184, loss: 0.726562
step: 184, loss: 0.531250
step: 184, loss: 0.632812
step: 184, loss: 0.546875
Train Step 185, bs: 16, loss: 0.66566, lr: 4e-05 final_score: 0.91594, mc_score: 0.67731, time: 545.94368
Train Step 185, bs: 16, loss: 0.65629, lr: 4e-05 final_score: 0.92207, mc_score: 0.68915, time: 547.11358
Train Step 185, bs: 16, loss: 0.66824, lr: 4e-05 final_score: 0.91344, mc_score: 0.67326, time: 546.19333
Train Step 185, bs: 16, loss: 0.67655, lr: 4e-05 final_score: 0.90706, mc_score: 0.65404, time: 545.28465
Train Step 185, bs: 16, loss: 0.65533, lr: 4e-05 final_score: 0.92037, mc_score: 0.69009, time: 600.75159
Train Step 185, bs: 16, loss: 0.65149, lr: 4e-05 final_score: 0.92345, mc_score: 0.69189, time: 549.42262
Train Step 185, bs: 16, loss: 0.67341, lr: 4e-05 final_score: 0.90808, mc_score: 0.65713, time: 549.74656
Train Step 185, bs: 16, loss: 0.64534, lr: 4e-05 final_score: 0.92696, mc_score: 0.70532, time: 561.41290
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 185, loss: 0.640625
step: 185, loss: 0.625000
step: 185, loss: 0.628906
step: 185, loss: 0.625000
step: 185, loss: 0.593750
step: 185, loss: 0.652344
step: 185, loss: 0.613281
step: 185, loss: 0.710938
Train Step 186, bs: 16, loss: 0.66552, lr: 4e-05 final_score: 0.91582, mc_score: 0.67840, time: 548.11432
Train Step 186, bs: 16, loss: 0.64523, lr: 4e-05 final_score: 0.92697, mc_score: 0.70551, time: 563.47076
Train Step 186, bs: 16, loss: 0.67315, lr: 4e-05 final_score: 0.90822, mc_score: 0.65840, time: 551.83083
Train Step 186, bs: 16, loss: 0.66784, lr: 4e-05 final_score: 0.91364, mc_score: 0.67435, time: 548.36246
Train Step 186, bs: 16, loss: 0.65181, lr: 4e-05 final_score: 0.92327, mc_score: 0.69220, time: 551.55771
Train Step 186, bs: 16, loss: 0.65614, lr: 4e-05 final_score: 0.92204, mc_score: 0.68882, time: 549.28431
Train Step 186, bs: 16, loss: 0.65532, lr: 4e-05 final_score: 0.92036, mc_score: 0.68906, time: 602.91913
Train Step 186, bs: 16, loss: 0.67621, lr: 4e-05 final_score: 0.90722, mc_score: 0.65386, time: 547.45274
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 186, loss: 0.523438
step: 186, loss: 0.632812
step: 186, loss: 0.597656
step: 186, loss: 0.609375
step: 186, loss: 0.515625
step: 186, loss: 0.578125
step: 186, loss: 0.734375
step: 186, loss: 0.648438
Train Step 187, bs: 16, loss: 0.66753, lr: 4e-05 final_score: 0.91368, mc_score: 0.67541, time: 550.37112
Train Step 187, bs: 16, loss: 0.65457, lr: 4e-05 final_score: 0.92079, mc_score: 0.69074, time: 604.92741
Train Step 187, bs: 16, loss: 0.67274, lr: 4e-05 final_score: 0.90837, mc_score: 0.65825, time: 553.83985
Train Step 187, bs: 16, loss: 0.67569, lr: 4e-05 final_score: 0.90745, mc_score: 0.65431, time: 549.46171
Train Step 187, bs: 16, loss: 0.64458, lr: 4e-05 final_score: 0.92724, mc_score: 0.70708, time: 565.47927
Train Step 187, bs: 16, loss: 0.66589, lr: 4e-05 final_score: 0.91547, mc_score: 0.67814, time: 550.15409
Train Step 187, bs: 16, loss: 0.65602, lr: 4e-05 final_score: 0.92196, mc_score: 0.68980, time: 551.31612
Train Step 187, bs: 16, loss: 0.65179, lr: 4e-05 final_score: 0.92310, mc_score: 0.69242, time: 553.62069
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 187, loss: 0.546875
step: 187, loss: 0.554688
step: 187, loss: 0.632812
step: 187, loss: 0.523438
step: 187, loss: 0.757812
step: 187, loss: 0.695312
step: 187, loss: 0.601562
step: 187, loss: 0.726562
Train Step 188, bs: 16, loss: 0.64394, lr: 4e-05 final_score: 0.92767, mc_score: 0.70863, time: 567.55872
Train Step 188, bs: 16, loss: 0.67319, lr: 4e-05 final_score: 0.90794, mc_score: 0.65738, time: 555.91834
Train Step 188, bs: 16, loss: 0.67546, lr: 4e-05 final_score: 0.90757, mc_score: 0.65488, time: 551.54086
Train Step 188, bs: 16, loss: 0.65404, lr: 4e-05 final_score: 0.92106, mc_score: 0.69103, time: 607.00778
Train Step 188, bs: 16, loss: 0.66526, lr: 4e-05 final_score: 0.91583, mc_score: 0.67916, time: 552.21781
Train Step 188, bs: 16, loss: 0.66768, lr: 4e-05 final_score: 0.91351, mc_score: 0.67512, time: 552.47288
Train Step 188, bs: 16, loss: 0.65573, lr: 4e-05 final_score: 0.92198, mc_score: 0.68870, time: 553.46418
Train Step 188, bs: 16, loss: 0.65219, lr: 4e-05 final_score: 0.92274, mc_score: 0.69065, time: 555.74490
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 188, loss: 0.570312
step: 188, loss: 0.640625
step: 188, loss: 0.597656
step: 188, loss: 0.773438
step: 188, loss: 0.535156
step: 188, loss: 0.574219
step: 188, loss: 0.562500
step: 188, loss: 0.632812
Train Step 189, bs: 16, loss: 0.66475, lr: 4e-05 final_score: 0.91600, mc_score: 0.67947, time: 554.33291
Train Step 189, bs: 16, loss: 0.64392, lr: 4e-05 final_score: 0.92762, mc_score: 0.70949, time: 569.68940
Train Step 189, bs: 16, loss: 0.67598, lr: 4e-05 final_score: 0.90710, mc_score: 0.65405, time: 553.67200
Train Step 189, bs: 16, loss: 0.66731, lr: 4e-05 final_score: 0.91358, mc_score: 0.67547, time: 554.58430
Train Step 189, bs: 16, loss: 0.67246, lr: 4e-05 final_score: 0.90843, mc_score: 0.65856, time: 558.05991
Train Step 189, bs: 16, loss: 0.65361, lr: 4e-05 final_score: 0.92141, mc_score: 0.69202, time: 609.14864
Train Step 189, bs: 16, loss: 0.65209, lr: 4e-05 final_score: 0.92284, mc_score: 0.69097, time: 557.80119
Train Step 189, bs: 16, loss: 0.65524, lr: 4e-05 final_score: 0.92229, mc_score: 0.68907, time: 555.54186
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 189, loss: 0.578125
step: 189, loss: 0.492188
step: 189, loss: 0.734375
step: 189, loss: 0.546875
step: 189, loss: 0.617188
step: 189, loss: 0.511719
step: 189, loss: 0.546875
step: 189, loss: 0.597656
Train Step 190, bs: 16, loss: 0.65483, lr: 4e-05 final_score: 0.92250, mc_score: 0.68940, time: 557.60200
Train Step 190, bs: 16, loss: 0.66385, lr: 4e-05 final_score: 0.91665, mc_score: 0.68115, time: 556.44678
Train Step 190, bs: 16, loss: 0.64440, lr: 4e-05 final_score: 0.92717, mc_score: 0.70903, time: 571.80494
Train Step 190, bs: 16, loss: 0.66667, lr: 4e-05 final_score: 0.91398, mc_score: 0.67715, time: 556.69619
Train Step 190, bs: 16, loss: 0.67217, lr: 4e-05 final_score: 0.90861, mc_score: 0.65967, time: 560.16549
Train Step 190, bs: 16, loss: 0.65180, lr: 4e-05 final_score: 0.92304, mc_score: 0.69130, time: 559.89671
Train Step 190, bs: 16, loss: 0.65287, lr: 4e-05 final_score: 0.92197, mc_score: 0.69364, time: 611.29825
Train Step 190, bs: 16, loss: 0.67530, lr: 4e-05 final_score: 0.90776, mc_score: 0.65519, time: 555.83218
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 190, loss: 0.531250
step: 190, loss: 0.757812
step: 190, loss: 0.498047
step: 190, loss: 0.632812
step: 190, loss: 0.527344
step: 190, loss: 0.593750
step: 190, loss: 0.601562
step: 190, loss: 0.515625
Train Step 191, bs: 16, loss: 0.67141, lr: 4e-05 final_score: 0.90921, mc_score: 0.66141, time: 562.23567
Train Step 191, bs: 16, loss: 0.65154, lr: 4e-05 final_score: 0.92312, mc_score: 0.69157, time: 561.96424
Train Step 191, bs: 16, loss: 0.67446, lr: 4e-05 final_score: 0.90832, mc_score: 0.65698, time: 557.85864
Train Step 191, bs: 16, loss: 0.66629, lr: 4e-05 final_score: 0.91438, mc_score: 0.67816, time: 558.76846
Train Step 191, bs: 16, loss: 0.65206, lr: 4e-05 final_score: 0.92254, mc_score: 0.69526, time: 613.32589
Train Step 191, bs: 16, loss: 0.64434, lr: 4e-05 final_score: 0.92724, mc_score: 0.70987, time: 573.87674
Train Step 191, bs: 16, loss: 0.66315, lr: 4e-05 final_score: 0.91706, mc_score: 0.68214, time: 558.52006
Train Step 191, bs: 16, loss: 0.65537, lr: 4e-05 final_score: 0.92206, mc_score: 0.68906, time: 559.70827
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 191, loss: 0.628906
step: 191, loss: 0.498047
step: 191, loss: 0.562500
step: 191, loss: 0.546875
step: 191, loss: 0.507812
step: 191, loss: 0.664062
step: 191, loss: 0.558594
step: 191, loss: 0.878906
Train Step 192, bs: 16, loss: 0.64426, lr: 4e-05 final_score: 0.92746, mc_score: 0.71007, time: 575.92521
Train Step 192, bs: 16, loss: 0.67138, lr: 4e-05 final_score: 0.90923, mc_score: 0.66111, time: 564.28501
Train Step 192, bs: 16, loss: 0.66575, lr: 4e-05 final_score: 0.91478, mc_score: 0.67858, time: 560.81701
Train Step 192, bs: 16, loss: 0.65455, lr: 4e-05 final_score: 0.92276, mc_score: 0.69068, time: 561.73979
Train Step 192, bs: 16, loss: 0.67359, lr: 4e-05 final_score: 0.90906, mc_score: 0.65875, time: 559.91285
Train Step 192, bs: 16, loss: 0.65105, lr: 4e-05 final_score: 0.92344, mc_score: 0.69250, time: 564.02400
Train Step 192, bs: 16, loss: 0.65151, lr: 4e-05 final_score: 0.92292, mc_score: 0.69619, time: 615.38668
Train Step 192, bs: 16, loss: 0.66428, lr: 4e-05 final_score: 0.91651, mc_score: 0.68192, time: 560.58549
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 192, loss: 0.593750
step: 192, loss: 0.550781
step: 192, loss: 0.558594
step: 192, loss: 0.546875
step: 192, loss: 0.656250
step: 192, loss: 0.474609
step: 192, loss: 0.531250
step: 192, loss: 0.671875
Train Step 193, bs: 16, loss: 0.64377, lr: 4e-05 final_score: 0.92788, mc_score: 0.71090, time: 577.97476
Train Step 193, bs: 16, loss: 0.65424, lr: 4e-05 final_score: 0.92296, mc_score: 0.69033, time: 563.78777
Train Step 193, bs: 16, loss: 0.66520, lr: 4e-05 final_score: 0.91538, mc_score: 0.68025, time: 562.86662
Train Step 193, bs: 16, loss: 0.67294, lr: 4e-05 final_score: 0.90961, mc_score: 0.65984, time: 561.95810
Train Step 193, bs: 16, loss: 0.66423, lr: 4e-05 final_score: 0.91654, mc_score: 0.68227, time: 562.62723
Train Step 193, bs: 16, loss: 0.65116, lr: 4e-05 final_score: 0.92345, mc_score: 0.69278, time: 566.07049
Train Step 193, bs: 16, loss: 0.65059, lr: 4e-05 final_score: 0.92360, mc_score: 0.69777, time: 617.43175
Train Step 193, bs: 16, loss: 0.67065, lr: 4e-05 final_score: 0.90995, mc_score: 0.66218, time: 566.34522
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 193, loss: 0.498047
step: 193, loss: 0.718750
step: 193, loss: 0.566406
step: 193, loss: 0.640625
step: 193, loss: 0.789062
step: 193, loss: 0.515625
step: 193, loss: 0.546875
step: 193, loss: 0.593750
Train Step 194, bs: 16, loss: 0.66338, lr: 4e-05 final_score: 0.91717, mc_score: 0.68329, time: 564.65791
Train Step 194, bs: 16, loss: 0.64416, lr: 4e-05 final_score: 0.92769, mc_score: 0.71110, time: 580.01488
Train Step 194, bs: 16, loss: 0.67213, lr: 4e-05 final_score: 0.91027, mc_score: 0.66158, time: 563.99596
Train Step 194, bs: 16, loss: 0.67011, lr: 4e-05 final_score: 0.91044, mc_score: 0.66334, time: 568.37525
Train Step 194, bs: 16, loss: 0.66483, lr: 4e-05 final_score: 0.91578, mc_score: 0.67999, time: 564.90690
Train Step 194, bs: 16, loss: 0.65062, lr: 4e-05 final_score: 0.92396, mc_score: 0.69435, time: 568.10233
Train Step 194, bs: 16, loss: 0.65417, lr: 4e-05 final_score: 0.92304, mc_score: 0.69005, time: 565.82976
Train Step 194, bs: 16, loss: 0.65131, lr: 4e-05 final_score: 0.92313, mc_score: 0.69743, time: 619.46458
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 194, loss: 0.640625
step: 194, loss: 0.570312
step: 194, loss: 0.906250
step: 194, loss: 0.617188
step: 194, loss: 0.656250
step: 194, loss: 0.527344
step: 194, loss: 0.480469
step: 194, loss: 0.687500
Train Step 195, bs: 16, loss: 0.66326, lr: 4e-05 final_score: 0.91732, mc_score: 0.68363, time: 566.67182
Train Step 195, bs: 16, loss: 0.66960, lr: 4e-05 final_score: 0.91086, mc_score: 0.66437, time: 570.38869
Train Step 195, bs: 16, loss: 0.64550, lr: 4e-05 final_score: 0.92671, mc_score: 0.71070, time: 582.02846
Train Step 195, bs: 16, loss: 0.65043, lr: 4e-05 final_score: 0.92384, mc_score: 0.69897, time: 621.47692
Train Step 195, bs: 16, loss: 0.64999, lr: 4e-05 final_score: 0.92442, mc_score: 0.69459, time: 570.11587
Train Step 195, bs: 16, loss: 0.66478, lr: 4e-05 final_score: 0.91594, mc_score: 0.68038, time: 566.92367
Train Step 195, bs: 16, loss: 0.65398, lr: 4e-05 final_score: 0.92325, mc_score: 0.69036, time: 567.85903
Train Step 195, bs: 16, loss: 0.67221, lr: 4e-05 final_score: 0.91023, mc_score: 0.66143, time: 566.07342
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 195, loss: 0.589844
step: 195, loss: 0.601562
step: 195, loss: 0.652344
step: 195, loss: 0.695312
step: 195, loss: 0.632812
step: 195, loss: 0.667969
step: 195, loss: 0.500000
step: 195, loss: 0.582031
Train Step 196, bs: 16, loss: 0.64554, lr: 4e-05 final_score: 0.92673, mc_score: 0.71151, time: 584.12309
Train Step 196, bs: 16, loss: 0.65012, lr: 4e-05 final_score: 0.92409, mc_score: 0.69926, time: 623.57071
Train Step 196, bs: 16, loss: 0.66973, lr: 4e-05 final_score: 0.91078, mc_score: 0.66472, time: 572.48487
Train Step 196, bs: 16, loss: 0.66329, lr: 4e-05 final_score: 0.91734, mc_score: 0.68462, time: 568.77015
Train Step 196, bs: 16, loss: 0.64974, lr: 4e-05 final_score: 0.92458, mc_score: 0.69552, time: 572.21177
Train Step 196, bs: 16, loss: 0.65387, lr: 4e-05 final_score: 0.92338, mc_score: 0.69067, time: 569.93973
Train Step 196, bs: 16, loss: 0.66394, lr: 4e-05 final_score: 0.91649, mc_score: 0.68199, time: 569.09643
Train Step 196, bs: 16, loss: 0.67175, lr: 4e-05 final_score: 0.91067, mc_score: 0.66254, time: 568.18647
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 196, loss: 0.593750
step: 196, loss: 0.531250
step: 196, loss: 0.761719
step: 196, loss: 0.640625
step: 196, loss: 0.632812
step: 196, loss: 0.675781
step: 196, loss: 0.554688
step: 196, loss: 0.503906
Train Step 197, bs: 16, loss: 0.67135, lr: 4e-05 final_score: 0.91080, mc_score: 0.66363, time: 570.21493
Train Step 197, bs: 16, loss: 0.66262, lr: 4e-05 final_score: 0.91775, mc_score: 0.68562, time: 570.89413
Train Step 197, bs: 16, loss: 0.65003, lr: 4e-05 final_score: 0.92412, mc_score: 0.69951, time: 625.69822
Train Step 197, bs: 16, loss: 0.64551, lr: 4e-05 final_score: 0.92660, mc_score: 0.71171, time: 586.25150
Train Step 197, bs: 16, loss: 0.65442, lr: 4e-05 final_score: 0.92297, mc_score: 0.69034, time: 572.06414
Train Step 197, bs: 16, loss: 0.64988, lr: 4e-05 final_score: 0.92440, mc_score: 0.69579, time: 574.33794
Train Step 197, bs: 16, loss: 0.66339, lr: 4e-05 final_score: 0.91678, mc_score: 0.68296, time: 571.15706
Train Step 197, bs: 16, loss: 0.66889, lr: 4e-05 final_score: 0.91121, mc_score: 0.66638, time: 574.62602
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 197, loss: 0.566406
step: 197, loss: 0.535156
step: 197, loss: 0.589844
step: 197, loss: 0.546875
step: 197, loss: 0.562500
step: 197, loss: 0.570312
step: 197, loss: 0.554688
step: 197, loss: 0.535156
Train Step 198, bs: 16, loss: 0.66213, lr: 4e-05 final_score: 0.91801, mc_score: 0.68659, time: 572.93730
Train Step 198, bs: 16, loss: 0.65409, lr: 4e-05 final_score: 0.92312, mc_score: 0.69185, time: 574.10657
Train Step 198, bs: 16, loss: 0.64496, lr: 4e-05 final_score: 0.92694, mc_score: 0.71317, time: 588.29418
Train Step 198, bs: 16, loss: 0.64943, lr: 4e-05 final_score: 0.92462, mc_score: 0.69730, time: 576.38049
Train Step 198, bs: 16, loss: 0.64951, lr: 4e-05 final_score: 0.92444, mc_score: 0.70039, time: 627.74204
Train Step 198, bs: 16, loss: 0.67084, lr: 4e-05 final_score: 0.91101, mc_score: 0.66533, time: 572.27594
Train Step 198, bs: 16, loss: 0.66274, lr: 4e-05 final_score: 0.91724, mc_score: 0.68455, time: 573.19915
Train Step 198, bs: 16, loss: 0.66831, lr: 4e-05 final_score: 0.91163, mc_score: 0.66743, time: 576.66961
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 198, loss: 0.648438
step: 198, loss: 0.531250
step: 198, loss: 0.703125
step: 198, loss: 0.585938
step: 198, loss: 0.531250
step: 198, loss: 0.566406
step: 198, loss: 0.695312
step: 198, loss: 0.679688
Train Step 199, bs: 16, loss: 0.64919, lr: 4e-05 final_score: 0.92462, mc_score: 0.70066, time: 629.75871
Train Step 199, bs: 16, loss: 0.66147, lr: 4e-05 final_score: 0.91826, mc_score: 0.68751, time: 574.95552
Train Step 199, bs: 16, loss: 0.67100, lr: 4e-05 final_score: 0.91090, mc_score: 0.66449, time: 574.29315
Train Step 199, bs: 16, loss: 0.66290, lr: 4e-05 final_score: 0.91710, mc_score: 0.68491, time: 575.24254
Train Step 199, bs: 16, loss: 0.66837, lr: 4e-05 final_score: 0.91188, mc_score: 0.66799, time: 578.71527
Train Step 199, bs: 16, loss: 0.64884, lr: 4e-05 final_score: 0.92486, mc_score: 0.69879, time: 578.44479
Train Step 199, bs: 16, loss: 0.65406, lr: 4e-05 final_score: 0.92306, mc_score: 0.69148, time: 576.17029
Train Step 199, bs: 16, loss: 0.64456, lr: 4e-05 final_score: 0.92717, mc_score: 0.71400, time: 590.36386
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 199, loss: 0.574219
step: 199, loss: 0.695312
step: 199, loss: 0.734375
step: 199, loss: 0.554688
step: 199, loss: 0.773438
step: 199, loss: 0.554688
step: 199, loss: 0.601562
step: 199, loss: 0.566406
Train Step 200, bs: 16, loss: 0.64942, lr: 4e-05 final_score: 0.92452, mc_score: 0.70151, time: 631.84197
Train Step 200, bs: 16, loss: 0.64411, lr: 4e-05 final_score: 0.92757, mc_score: 0.71542, time: 592.39824
Train Step 200, bs: 16, loss: 0.67132, lr: 4e-05 final_score: 0.91060, mc_score: 0.66355, time: 576.37888
Train Step 200, bs: 16, loss: 0.66780, lr: 4e-05 final_score: 0.91218, mc_score: 0.66887, time: 580.76404
Train Step 200, bs: 16, loss: 0.66104, lr: 4e-05 final_score: 0.91860, mc_score: 0.68776, time: 577.05271
Train Step 200, bs: 16, loss: 0.66346, lr: 4e-05 final_score: 0.91662, mc_score: 0.68398, time: 577.30031
Train Step 200, bs: 16, loss: 0.64860, lr: 4e-05 final_score: 0.92493, mc_score: 0.69966, time: 580.53936
Train Step 200, bs: 16, loss: 0.65362, lr: 4e-05 final_score: 0.92334, mc_score: 0.69177, time: 578.27552
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 200, loss: 0.695312
step: 200, loss: 0.589844
step: 200, loss: 0.683594
step: 200, loss: 0.539062
step: 200, loss: 0.710938
step: 200, loss: 0.753906
step: 200, loss: 0.742188
step: 200, loss: 0.664062
Train Step 201, bs: 16, loss: 0.66121, lr: 4e-05 final_score: 0.91841, mc_score: 0.68673, time: 579.13685
Train Step 201, bs: 16, loss: 0.64384, lr: 4e-05 final_score: 0.92792, mc_score: 0.71624, time: 594.49501
Train Step 201, bs: 16, loss: 0.64959, lr: 4e-05 final_score: 0.92439, mc_score: 0.70115, time: 633.94250
Train Step 201, bs: 16, loss: 0.67066, lr: 4e-05 final_score: 0.91099, mc_score: 0.66522, time: 578.47711
Train Step 201, bs: 16, loss: 0.66802, lr: 4e-05 final_score: 0.91201, mc_score: 0.66862, time: 582.86489
Train Step 201, bs: 16, loss: 0.66391, lr: 4e-05 final_score: 0.91618, mc_score: 0.68426, time: 579.40423
Train Step 201, bs: 16, loss: 0.65406, lr: 4e-05 final_score: 0.92289, mc_score: 0.69138, time: 580.35520
Train Step 201, bs: 16, loss: 0.64868, lr: 4e-05 final_score: 0.92489, mc_score: 0.69928, time: 582.63402
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 201, loss: 0.765625
step: 201, loss: 0.574219
step: 201, loss: 0.570312
step: 201, loss: 0.609375
step: 201, loss: 0.773438
step: 201, loss: 0.562500
step: 201, loss: 0.558594
step: 201, loss: 0.546875
Train Step 202, bs: 16, loss: 0.66172, lr: 4e-05 final_score: 0.91802, mc_score: 0.68640, time: 581.22149
Train Step 202, bs: 16, loss: 0.64930, lr: 4e-05 final_score: 0.92434, mc_score: 0.69887, time: 584.66430
Train Step 202, bs: 16, loss: 0.67016, lr: 4e-05 final_score: 0.91131, mc_score: 0.66624, time: 580.55907
Train Step 202, bs: 16, loss: 0.64939, lr: 4e-05 final_score: 0.92458, mc_score: 0.70200, time: 636.02557
Train Step 202, bs: 16, loss: 0.65367, lr: 4e-05 final_score: 0.92310, mc_score: 0.69224, time: 582.39224
Train Step 202, bs: 16, loss: 0.64344, lr: 4e-05 final_score: 0.92809, mc_score: 0.71704, time: 596.58379
Train Step 202, bs: 16, loss: 0.66748, lr: 4e-05 final_score: 0.91239, mc_score: 0.66899, time: 584.94454
Train Step 202, bs: 16, loss: 0.66333, lr: 4e-05 final_score: 0.91655, mc_score: 0.68523, time: 581.48126
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 202, loss: 0.585938
step: 202, loss: 0.640625
step: 202, loss: 0.511719
step: 202, loss: 0.765625
step: 202, loss: 0.531250
step: 202, loss: 0.593750
step: 202, loss: 0.636719
step: 202, loss: 0.640625
Train Step 203, bs: 16, loss: 0.64935, lr: 4e-05 final_score: 0.92467, mc_score: 0.70222, time: 638.06963
Train Step 203, bs: 16, loss: 0.66135, lr: 4e-05 final_score: 0.91835, mc_score: 0.68670, time: 583.26517
Train Step 203, bs: 16, loss: 0.66938, lr: 4e-05 final_score: 0.91188, mc_score: 0.66788, time: 582.60331
Train Step 203, bs: 16, loss: 0.65422, lr: 4e-05 final_score: 0.92265, mc_score: 0.69131, time: 584.43743
Train Step 203, bs: 16, loss: 0.64319, lr: 4e-05 final_score: 0.92822, mc_score: 0.71779, time: 598.62322
Train Step 203, bs: 16, loss: 0.66320, lr: 4e-05 final_score: 0.91661, mc_score: 0.68558, time: 583.52449
Train Step 203, bs: 16, loss: 0.64872, lr: 4e-05 final_score: 0.92466, mc_score: 0.69973, time: 586.70923
Train Step 203, bs: 16, loss: 0.66734, lr: 4e-05 final_score: 0.91262, mc_score: 0.67006, time: 587.00673
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 203, loss: 0.488281
step: 203, loss: 0.652344
step: 203, loss: 0.531250
step: 203, loss: 0.515625
step: 203, loss: 0.515625
step: 203, loss: 0.488281
step: 203, loss: 0.519531
step: 203, loss: 0.500000
Train Step 204, bs: 16, loss: 0.66050, lr: 4e-05 final_score: 0.91890, mc_score: 0.68825, time: 585.31432
Train Step 204, bs: 16, loss: 0.64873, lr: 4e-05 final_score: 0.92470, mc_score: 0.69987, time: 588.75707
Train Step 204, bs: 16, loss: 0.66871, lr: 4e-05 final_score: 0.91249, mc_score: 0.66891, time: 584.65167
Train Step 204, bs: 16, loss: 0.64257, lr: 4e-05 final_score: 0.92880, mc_score: 0.71918, time: 600.67139
Train Step 204, bs: 16, loss: 0.64856, lr: 4e-05 final_score: 0.92534, mc_score: 0.70368, time: 640.12041
Train Step 204, bs: 16, loss: 0.65354, lr: 4e-05 final_score: 0.92311, mc_score: 0.69278, time: 586.48574
Train Step 204, bs: 16, loss: 0.66652, lr: 4e-05 final_score: 0.91333, mc_score: 0.67160, time: 589.04441
Train Step 204, bs: 16, loss: 0.66249, lr: 4e-05 final_score: 0.91708, mc_score: 0.68649, time: 585.57610
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 204, loss: 0.777344
step: 204, loss: 0.679688
step: 204, loss: 0.486328
step: 204, loss: 0.468750
step: 204, loss: 0.515625
step: 204, loss: 0.496094
step: 204, loss: 0.738281
step: 204, loss: 0.726562
Train Step 205, bs: 16, loss: 0.66107, lr: 4e-05 final_score: 0.91854, mc_score: 0.68858, time: 587.33857
Train Step 205, bs: 16, loss: 0.66876, lr: 4e-05 final_score: 0.91243, mc_score: 0.66936, time: 586.67564
Train Step 205, bs: 16, loss: 0.64181, lr: 4e-05 final_score: 0.92924, mc_score: 0.72055, time: 602.71569
Train Step 205, bs: 16, loss: 0.65264, lr: 4e-05 final_score: 0.92379, mc_score: 0.69433, time: 588.52933
Train Step 205, bs: 16, loss: 0.64791, lr: 4e-05 final_score: 0.92580, mc_score: 0.70512, time: 642.16461
Train Step 205, bs: 16, loss: 0.64799, lr: 4e-05 final_score: 0.92524, mc_score: 0.70138, time: 590.80888
Train Step 205, bs: 16, loss: 0.66286, lr: 4e-05 final_score: 0.91683, mc_score: 0.68620, time: 587.62970
Train Step 205, bs: 16, loss: 0.66682, lr: 4e-05 final_score: 0.91304, mc_score: 0.67198, time: 591.10021
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 205, loss: 0.593750
step: 205, loss: 0.621094
step: 205, loss: 0.519531
step: 205, loss: 0.570312
step: 205, loss: 0.585938
step: 205, loss: 0.562500
step: 205, loss: 0.515625
step: 205, loss: 0.671875
Train Step 206, bs: 16, loss: 0.64171, lr: 4e-05 final_score: 0.92944, mc_score: 0.72126, time: 604.77429
Train Step 206, bs: 16, loss: 0.66804, lr: 4e-05 final_score: 0.91302, mc_score: 0.67094, time: 588.75547
Train Step 206, bs: 16, loss: 0.66074, lr: 4e-05 final_score: 0.91879, mc_score: 0.68950, time: 589.42181
Train Step 206, bs: 16, loss: 0.65224, lr: 4e-05 final_score: 0.92413, mc_score: 0.69523, time: 590.58985
Train Step 206, bs: 16, loss: 0.66249, lr: 4e-05 final_score: 0.91711, mc_score: 0.68650, time: 589.67426
Train Step 206, bs: 16, loss: 0.64757, lr: 4e-05 final_score: 0.92561, mc_score: 0.70229, time: 592.87134
Train Step 206, bs: 16, loss: 0.64727, lr: 4e-05 final_score: 0.92624, mc_score: 0.70656, time: 644.23294
Train Step 206, bs: 16, loss: 0.66684, lr: 4e-05 final_score: 0.91297, mc_score: 0.67235, time: 593.15339
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 206, loss: 0.515625
step: 206, loss: 0.578125
step: 206, loss: 0.585938
step: 206, loss: 0.578125
step: 206, loss: 0.507812
step: 206, loss: 0.671875
step: 206, loss: 0.593750
step: 206, loss: 0.703125
Train Step 207, bs: 16, loss: 0.64694, lr: 4e-05 final_score: 0.92606, mc_score: 0.70317, time: 594.89218
Train Step 207, bs: 16, loss: 0.66726, lr: 4e-05 final_score: 0.91350, mc_score: 0.67191, time: 590.80427
Train Step 207, bs: 16, loss: 0.65188, lr: 4e-05 final_score: 0.92429, mc_score: 0.69611, time: 592.63708
Train Step 207, bs: 16, loss: 0.64144, lr: 4e-05 final_score: 0.92956, mc_score: 0.72200, time: 606.82395
Train Step 207, bs: 16, loss: 0.66034, lr: 4e-05 final_score: 0.91917, mc_score: 0.69037, time: 591.46751
Train Step 207, bs: 16, loss: 0.64739, lr: 4e-05 final_score: 0.92624, mc_score: 0.70679, time: 646.27168
Train Step 207, bs: 16, loss: 0.66702, lr: 4e-05 final_score: 0.91275, mc_score: 0.67275, time: 595.19940
Train Step 207, bs: 16, loss: 0.66216, lr: 4e-05 final_score: 0.91730, mc_score: 0.68741, time: 591.74106
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 207, loss: 0.566406
step: 207, loss: 0.472656
step: 207, loss: 0.722656
step: 207, loss: 0.503906
step: 207, loss: 0.558594
step: 207, loss: 0.480469
step: 207, loss: 0.507812
step: 207, loss: 0.562500
Train Step 208, bs: 16, loss: 0.64062, lr: 4e-05 final_score: 0.93013, mc_score: 0.72333, time: 608.87545
Train Step 208, bs: 16, loss: 0.66648, lr: 4e-05 final_score: 0.91415, mc_score: 0.67348, time: 592.85670
Train Step 208, bs: 16, loss: 0.65222, lr: 4e-05 final_score: 0.92408, mc_score: 0.69638, time: 594.68951
Train Step 208, bs: 16, loss: 0.66166, lr: 4e-05 final_score: 0.91767, mc_score: 0.68767, time: 593.77592
Train Step 208, bs: 16, loss: 0.64627, lr: 4e-05 final_score: 0.92655, mc_score: 0.70399, time: 596.98582
Train Step 208, bs: 16, loss: 0.64698, lr: 4e-05 final_score: 0.92658, mc_score: 0.70700, time: 648.34864
Train Step 208, bs: 16, loss: 0.65989, lr: 4e-05 final_score: 0.91957, mc_score: 0.69070, time: 593.55908
Train Step 208, bs: 16, loss: 0.66612, lr: 4e-05 final_score: 0.91337, mc_score: 0.67431, time: 597.28654
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 208, loss: 0.470703
step: 208, loss: 0.578125
step: 208, loss: 0.500000
step: 208, loss: 0.535156
step: 208, loss: 0.507812
step: 208, loss: 0.523438
step: 208, loss: 0.542969
step: 208, loss: 0.488281
Train Step 209, bs: 16, loss: 0.66105, lr: 4e-05 final_score: 0.91809, mc_score: 0.68852, time: 595.84983
Train Step 209, bs: 16, loss: 0.66605, lr: 4e-05 final_score: 0.91444, mc_score: 0.67384, time: 594.94024
Train Step 209, bs: 16, loss: 0.63981, lr: 4e-05 final_score: 0.93061, mc_score: 0.72465, time: 610.95989
Train Step 209, bs: 16, loss: 0.65149, lr: 4e-05 final_score: 0.92456, mc_score: 0.69782, time: 596.77340
Train Step 209, bs: 16, loss: 0.65933, lr: 4e-05 final_score: 0.92009, mc_score: 0.69156, time: 595.61413
Train Step 209, bs: 16, loss: 0.64561, lr: 4e-05 final_score: 0.92705, mc_score: 0.70482, time: 599.05559
Train Step 209, bs: 16, loss: 0.64639, lr: 4e-05 final_score: 0.92717, mc_score: 0.70840, time: 650.41735
Train Step 209, bs: 16, loss: 0.66527, lr: 4e-05 final_score: 0.91395, mc_score: 0.67591, time: 599.33071
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 209, loss: 0.644531
step: 209, loss: 0.597656
step: 209, loss: 0.886719
step: 209, loss: 0.546875
step: 209, loss: 0.648438
step: 209, loss: 0.683594
step: 209, loss: 0.605469
step: 209, loss: 0.812500
Train Step 210, bs: 16, loss: 0.65926, lr: 4e-05 final_score: 0.92002, mc_score: 0.69183, time: 597.64473
Train Step 210, bs: 16, loss: 0.63937, lr: 4e-05 final_score: 0.93107, mc_score: 0.72597, time: 613.00196
Train Step 210, bs: 16, loss: 0.66632, lr: 4e-05 final_score: 0.91348, mc_score: 0.67512, time: 601.36120
Train Step 210, bs: 16, loss: 0.66075, lr: 4e-05 final_score: 0.91841, mc_score: 0.68937, time: 597.89288
Train Step 210, bs: 16, loss: 0.66597, lr: 4e-05 final_score: 0.91452, mc_score: 0.67358, time: 596.98333
Train Step 210, bs: 16, loss: 0.65165, lr: 4e-05 final_score: 0.92444, mc_score: 0.69866, time: 598.81723
Train Step 210, bs: 16, loss: 0.64640, lr: 4e-05 final_score: 0.92654, mc_score: 0.70507, time: 601.10787
Train Step 210, bs: 16, loss: 0.64620, lr: 4e-05 final_score: 0.92736, mc_score: 0.70860, time: 652.47106
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 210, loss: 0.480469
step: 210, loss: 0.566406
step: 210, loss: 0.746094
step: 210, loss: 0.558594
step: 210, loss: 0.515625
step: 210, loss: 0.667969
step: 210, loss: 0.472656
step: 210, loss: 0.710938
Train Step 211, bs: 16, loss: 0.65967, lr: 4e-05 final_score: 0.91969, mc_score: 0.69157, time: 599.67583
Train Step 211, bs: 16, loss: 0.63862, lr: 4e-05 final_score: 0.93162, mc_score: 0.72727, time: 615.03229
Train Step 211, bs: 16, loss: 0.66585, lr: 4e-05 final_score: 0.91380, mc_score: 0.67556, time: 603.39224
Train Step 211, bs: 16, loss: 0.66006, lr: 4e-05 final_score: 0.91880, mc_score: 0.69024, time: 599.92596
Train Step 211, bs: 16, loss: 0.66546, lr: 4e-05 final_score: 0.91494, mc_score: 0.67452, time: 599.01714
Train Step 211, bs: 16, loss: 0.64537, lr: 4e-05 final_score: 0.92784, mc_score: 0.70998, time: 654.51493
Train Step 211, bs: 16, loss: 0.64650, lr: 4e-05 final_score: 0.92640, mc_score: 0.70532, time: 603.15896
Train Step 211, bs: 16, loss: 0.65193, lr: 4e-05 final_score: 0.92422, mc_score: 0.69886, time: 600.89938
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 211, loss: 0.816406
step: 211, loss: 0.554688
step: 211, loss: 0.761719
step: 211, loss: 0.640625
step: 211, loss: 0.718750
step: 211, loss: 0.882812
step: 211, loss: 0.523438
step: 211, loss: 0.515625
Train Step 212, bs: 16, loss: 0.65147, lr: 4e-05 final_score: 0.92459, mc_score: 0.69970, time: 602.93516
Train Step 212, bs: 16, loss: 0.66041, lr: 4e-05 final_score: 0.91920, mc_score: 0.69014, time: 601.76684
Train Step 212, bs: 16, loss: 0.66591, lr: 4e-05 final_score: 0.91478, mc_score: 0.67492, time: 601.10459
Train Step 212, bs: 16, loss: 0.64535, lr: 4e-05 final_score: 0.92789, mc_score: 0.71074, time: 656.57197
Train Step 212, bs: 16, loss: 0.66034, lr: 4e-05 final_score: 0.91861, mc_score: 0.69046, time: 602.03364
Train Step 212, bs: 16, loss: 0.66687, lr: 4e-05 final_score: 0.91308, mc_score: 0.67476, time: 605.50719
Train Step 212, bs: 16, loss: 0.63807, lr: 4e-05 final_score: 0.93201, mc_score: 0.72794, time: 617.15187
Train Step 212, bs: 16, loss: 0.64589, lr: 4e-05 final_score: 0.92687, mc_score: 0.70612, time: 605.23814
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 212, loss: 0.570312
step: 212, loss: 0.792969
step: 212, loss: 0.566406
step: 212, loss: 0.667969
step: 212, loss: 0.470703
step: 212, loss: 0.527344
step: 212, loss: 0.730469
step: 212, loss: 0.667969
Train Step 213, bs: 16, loss: 0.65213, lr: 4e-05 final_score: 0.92405, mc_score: 0.69932, time: 605.00129
Train Step 213, bs: 16, loss: 0.65999, lr: 4e-05 final_score: 0.91945, mc_score: 0.69042, time: 603.83383
Train Step 213, bs: 16, loss: 0.64546, lr: 4e-05 final_score: 0.92776, mc_score: 0.71032, time: 658.63736
Train Step 213, bs: 16, loss: 0.63774, lr: 4e-05 final_score: 0.93221, mc_score: 0.72744, time: 619.19079
Train Step 213, bs: 16, loss: 0.65972, lr: 4e-05 final_score: 0.91911, mc_score: 0.69132, time: 604.09007
Train Step 213, bs: 16, loss: 0.66500, lr: 4e-05 final_score: 0.91534, mc_score: 0.67644, time: 603.18021
Train Step 213, bs: 16, loss: 0.64628, lr: 4e-05 final_score: 0.92665, mc_score: 0.70629, time: 607.28729
Train Step 213, bs: 16, loss: 0.66688, lr: 4e-05 final_score: 0.91315, mc_score: 0.67514, time: 607.56832
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 213, loss: 0.816406
step: 213, loss: 0.550781
step: 213, loss: 0.507812
step: 213, loss: 0.478516
step: 213, loss: 0.636719
step: 213, loss: 0.679688
step: 213, loss: 0.482422
step: 213, loss: 0.617188
Train Step 214, bs: 16, loss: 0.66072, lr: 4e-05 final_score: 0.91881, mc_score: 0.68943, time: 605.88737
Train Step 214, bs: 16, loss: 0.65166, lr: 4e-05 final_score: 0.92426, mc_score: 0.70013, time: 607.05586
Train Step 214, bs: 16, loss: 0.64481, lr: 4e-05 final_score: 0.92815, mc_score: 0.71168, time: 660.69090
Train Step 214, bs: 16, loss: 0.63699, lr: 4e-05 final_score: 0.93264, mc_score: 0.72873, time: 621.24449
Train Step 214, bs: 16, loss: 0.64644, lr: 4e-05 final_score: 0.92662, mc_score: 0.70647, time: 609.33163
Train Step 214, bs: 16, loss: 0.66487, lr: 4e-05 final_score: 0.91539, mc_score: 0.67561, time: 605.22708
Train Step 214, bs: 16, loss: 0.65889, lr: 4e-05 final_score: 0.91954, mc_score: 0.69278, time: 606.14436
Train Step 214, bs: 16, loss: 0.66665, lr: 4e-05 final_score: 0.91326, mc_score: 0.67548, time: 609.62551
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 214, loss: 0.554688
step: 214, loss: 0.496094
step: 214, loss: 0.617188
step: 214, loss: 0.574219
step: 214, loss: 0.617188
step: 214, loss: 0.753906
step: 214, loss: 0.640625
step: 214, loss: 0.593750
Train Step 215, bs: 16, loss: 0.66023, lr: 4e-05 final_score: 0.91911, mc_score: 0.68967, time: 607.94106
Train Step 215, bs: 16, loss: 0.63690, lr: 4e-05 final_score: 0.93270, mc_score: 0.72882, time: 623.29771
Train Step 215, bs: 16, loss: 0.66408, lr: 4e-05 final_score: 0.91590, mc_score: 0.67712, time: 607.27886
Train Step 215, bs: 16, loss: 0.65880, lr: 4e-05 final_score: 0.91959, mc_score: 0.69362, time: 608.19951
Train Step 215, bs: 16, loss: 0.65130, lr: 4e-05 final_score: 0.92449, mc_score: 0.70090, time: 609.11028
Train Step 215, bs: 16, loss: 0.66631, lr: 4e-05 final_score: 0.91357, mc_score: 0.67649, time: 611.67371
Train Step 215, bs: 16, loss: 0.64630, lr: 4e-05 final_score: 0.92679, mc_score: 0.70611, time: 611.43512
Train Step 215, bs: 16, loss: 0.64532, lr: 4e-05 final_score: 0.92765, mc_score: 0.71067, time: 662.79832
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 215, loss: 0.582031
step: 215, loss: 0.664062
step: 215, loss: 0.617188
step: 215, loss: 0.683594
step: 215, loss: 0.714844
step: 215, loss: 0.695312
step: 215, loss: 0.550781
step: 215, loss: 0.554688
Train Step 216, bs: 16, loss: 0.66386, lr: 4e-05 final_score: 0.91614, mc_score: 0.67801, time: 609.35934
Train Step 216, bs: 16, loss: 0.65987, lr: 4e-05 final_score: 0.91928, mc_score: 0.69052, time: 610.02363
Train Step 216, bs: 16, loss: 0.65906, lr: 4e-05 final_score: 0.91940, mc_score: 0.69331, time: 610.27906
Train Step 216, bs: 16, loss: 0.66644, lr: 4e-05 final_score: 0.91353, mc_score: 0.67753, time: 613.74849
Train Step 216, bs: 16, loss: 0.64488, lr: 4e-05 final_score: 0.92788, mc_score: 0.71144, time: 664.84832
Train Step 216, bs: 16, loss: 0.64588, lr: 4e-05 final_score: 0.92705, mc_score: 0.70695, time: 613.49211
Train Step 216, bs: 16, loss: 0.65136, lr: 4e-05 final_score: 0.92443, mc_score: 0.70114, time: 611.22914
Train Step 216, bs: 16, loss: 0.63712, lr: 4e-05 final_score: 0.93257, mc_score: 0.72776, time: 625.41706
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 216, loss: 0.589844
step: 216, loss: 0.511719
step: 216, loss: 0.617188
step: 216, loss: 0.656250
step: 216, loss: 0.570312
step: 216, loss: 0.466797
step: 216, loss: 0.527344
step: 216, loss: 0.648438
Train Step 217, bs: 16, loss: 0.64526, lr: 4e-05 final_score: 0.92743, mc_score: 0.70826, time: 615.53052
Train Step 217, bs: 16, loss: 0.65954, lr: 4e-05 final_score: 0.91947, mc_score: 0.69027, time: 612.08927
Train Step 217, bs: 16, loss: 0.63702, lr: 4e-05 final_score: 0.93259, mc_score: 0.72786, time: 627.44657
Train Step 217, bs: 16, loss: 0.65138, lr: 4e-05 final_score: 0.92431, mc_score: 0.70087, time: 613.25988
Train Step 217, bs: 16, loss: 0.65846, lr: 4e-05 final_score: 0.91977, mc_score: 0.69417, time: 612.34567
Train Step 217, bs: 16, loss: 0.66636, lr: 4e-05 final_score: 0.91353, mc_score: 0.67722, time: 615.81586
Train Step 217, bs: 16, loss: 0.66343, lr: 4e-05 final_score: 0.91640, mc_score: 0.67894, time: 611.47244
Train Step 217, bs: 16, loss: 0.64406, lr: 4e-05 final_score: 0.92839, mc_score: 0.71276, time: 666.94038
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 217, loss: 0.625000
step: 217, loss: 0.507812
step: 217, loss: 0.550781
step: 217, loss: 0.738281
step: 217, loss: 0.535156
step: 217, loss: 0.656250
step: 217, loss: 0.636719
step: 217, loss: 0.531250
Train Step 218, bs: 16, loss: 0.63643, lr: 4e-05 final_score: 0.93302, mc_score: 0.72911, time: 629.52318
Train Step 218, bs: 16, loss: 0.65939, lr: 4e-05 final_score: 0.91955, mc_score: 0.69050, time: 614.16777
Train Step 218, bs: 16, loss: 0.64364, lr: 4e-05 final_score: 0.92868, mc_score: 0.71239, time: 668.97092
Train Step 218, bs: 16, loss: 0.65178, lr: 4e-05 final_score: 0.92397, mc_score: 0.70111, time: 615.33785
Train Step 218, bs: 16, loss: 0.64522, lr: 4e-05 final_score: 0.92743, mc_score: 0.70850, time: 617.62194
Train Step 218, bs: 16, loss: 0.65789, lr: 4e-05 final_score: 0.92005, mc_score: 0.69501, time: 614.42711
Train Step 218, bs: 16, loss: 0.66340, lr: 4e-05 final_score: 0.91639, mc_score: 0.67927, time: 613.51748
Train Step 218, bs: 16, loss: 0.66574, lr: 4e-05 final_score: 0.91401, mc_score: 0.67864, time: 617.89680
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 218, loss: 0.589844
step: 218, loss: 0.722656
step: 218, loss: 0.488281
step: 218, loss: 0.542969
step: 218, loss: 0.726562
step: 218, loss: 0.855469
step: 218, loss: 0.535156
step: 218, loss: 0.781250
Train Step 219, bs: 16, loss: 0.65907, lr: 4e-05 final_score: 0.91979, mc_score: 0.69082, time: 616.20893
Train Step 219, bs: 16, loss: 0.65103, lr: 4e-05 final_score: 0.92445, mc_score: 0.70245, time: 617.37876
Train Step 219, bs: 16, loss: 0.63683, lr: 4e-05 final_score: 0.93267, mc_score: 0.72858, time: 631.56512
Train Step 219, bs: 16, loss: 0.66394, lr: 4e-05 final_score: 0.91597, mc_score: 0.67745, time: 615.56005
Train Step 219, bs: 16, loss: 0.66514, lr: 4e-05 final_score: 0.91432, mc_score: 0.68015, time: 619.93797
Train Step 219, bs: 16, loss: 0.65879, lr: 4e-05 final_score: 0.91948, mc_score: 0.69529, time: 616.47431
Train Step 219, bs: 16, loss: 0.64318, lr: 4e-05 final_score: 0.92901, mc_score: 0.71312, time: 671.05216
Train Step 219, bs: 16, loss: 0.64559, lr: 4e-05 final_score: 0.92710, mc_score: 0.70814, time: 619.70191
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 219, loss: 0.494141
step: 219, loss: 0.714844
step: 219, loss: 0.523438
step: 219, loss: 0.546875
step: 219, loss: 0.550781
step: 219, loss: 0.734375
step: 219, loss: 0.570312
step: 219, loss: 0.718750
Train Step 220, bs: 16, loss: 0.65832, lr: 4e-05 final_score: 0.92032, mc_score: 0.69222, time: 618.28768
Train Step 220, bs: 16, loss: 0.65132, lr: 4e-05 final_score: 0.92416, mc_score: 0.70212, time: 619.45595
Train Step 220, bs: 16, loss: 0.63631, lr: 4e-05 final_score: 0.93302, mc_score: 0.72922, time: 633.64422
Train Step 220, bs: 16, loss: 0.66341, lr: 4e-05 final_score: 0.91643, mc_score: 0.67836, time: 617.62533
Train Step 220, bs: 16, loss: 0.66462, lr: 4e-05 final_score: 0.91464, mc_score: 0.68159, time: 622.01806
Train Step 220, bs: 16, loss: 0.65914, lr: 4e-05 final_score: 0.91920, mc_score: 0.69495, time: 618.54939
Train Step 220, bs: 16, loss: 0.64284, lr: 4e-05 final_score: 0.92923, mc_score: 0.71386, time: 673.12930
Train Step 220, bs: 16, loss: 0.64593, lr: 4e-05 final_score: 0.92672, mc_score: 0.70835, time: 621.76917
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 220, loss: 0.550781
step: 220, loss: 0.582031
step: 220, loss: 0.519531
step: 220, loss: 0.507812
step: 220, loss: 0.585938
step: 220, loss: 0.470703
step: 220, loss: 0.574219
step: 220, loss: 0.511719
Train Step 221, bs: 16, loss: 0.65101, lr: 4e-05 final_score: 0.92437, mc_score: 0.70287, time: 621.50840
Train Step 221, bs: 16, loss: 0.65783, lr: 4e-05 final_score: 0.92056, mc_score: 0.69362, time: 620.34138
Train Step 221, bs: 16, loss: 0.63578, lr: 4e-05 final_score: 0.93331, mc_score: 0.72986, time: 635.71666
Train Step 221, bs: 16, loss: 0.64223, lr: 4e-05 final_score: 0.92957, mc_score: 0.71460, time: 675.16455
Train Step 221, bs: 16, loss: 0.66253, lr: 4e-05 final_score: 0.91689, mc_score: 0.67981, time: 619.70975
Train Step 221, bs: 16, loss: 0.66421, lr: 4e-05 final_score: 0.91494, mc_score: 0.68314, time: 624.08894
Train Step 221, bs: 16, loss: 0.65847, lr: 4e-05 final_score: 0.91955, mc_score: 0.69634, time: 620.62038
Train Step 221, bs: 16, loss: 0.64565, lr: 4e-05 final_score: 0.92706, mc_score: 0.70858, time: 623.81618
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
lr: 4e-05
step: 221, loss: 0.582031
step: 221, loss: 0.570312
step: 221, loss: 0.691406
step: 221, loss: 0.546875
step: 221, loss: 0.515625
step: 221, loss: 0.605469
step: 221, loss: 0.640625
step: 221, loss: 0.546875
Train Step 222, bs: 16, loss: 0.65749, lr: 4e-05 final_score: 0.92071, mc_score: 0.69383, time: 622.40808
Train Step 222, bs: 16, loss: 0.64207, lr: 4e-05 final_score: 0.92967, mc_score: 0.71533, time: 677.21190
Train Step 222, bs: 16, loss: 0.63549, lr: 4e-05 final_score: 0.93346, mc_score: 0.72995, time: 637.76491
Train Step 222, bs: 16, loss: 0.65119, lr: 4e-05 final_score: 0.92413, mc_score: 0.70190, time: 623.57726
Train Step 222, bs: 16, loss: 0.66368, lr: 4e-05 final_score: 0.91528, mc_score: 0.68403, time: 626.12430
Train Step 222, bs: 16, loss: 0.65782, lr: 4e-05 final_score: 0.92002, mc_score: 0.69772, time: 622.65585
Train Step 222, bs: 16, loss: 0.66243, lr: 4e-05 final_score: 0.91694, mc_score: 0.68012, time: 621.75954
Train Step 222, bs: 16, loss: 0.64521, lr: 4e-05 final_score: 0.92732, mc_score: 0.70987, time: 625.86533
```
```
Train Step 1, bs: 16, loss: 1.07812, lr: 4e-05 final_score: 0.60938, mc_score: 0.00000, time: 74.85108
Train Step 1, bs: 16, loss: 1.20312, lr: 4e-05 final_score: 0.56667, mc_score: -0.33333, time: 79.23281
Train Step 1, bs: 16, loss: 1.38281, lr: 4e-05 final_score: 0.29167, mc_score: -0.21822, time: 75.51826
Train Step 1, bs: 16, loss: 0.92188, lr: 4e-05 final_score: 0.80159, mc_score: 0.54470, time: 78.96233
Train Step 1, bs: 16, loss: 1.14844, lr: 4e-05 final_score: 0.59375, mc_score: -0.25820, time: 75.76754
Train Step 1, bs: 16, loss: 1.12500, lr: 4e-05 final_score: 0.33333, mc_score: -0.29277, time: 76.68898
Train Step 1, bs: 16, loss: 1.12500, lr: 4e-05 final_score: 0.43333, mc_score: -0.29277, time: 90.88104
Train Step 1, bs: 16, loss: 1.13281, lr: 4e-05 final_score: 0.48438, mc_score: 0.37796, time: 130.33304
```
```
[DEBUG]2020-06-19 09:25:13,427:utils:itr: 214, num_batch: 214, last loss: 1.234375, smooth_loss: 1.201096
[DEBUG]2020-06-19 09:25:13,417:utils:itr: 228, num_batch: 228, last loss: 1.210938, smooth_loss: 1.145556
[DEBUG]2020-06-19 09:25:13,675:utils:loss_avg: 1.16315, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.49119, mc_score:-0.01076
[DEBUG]2020-06-19 09:25:13,680:utils:loss_avg: 1.15451, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.48564, mc_score:-0.00425
[DEBUG]2020-06-19 09:25:14,201:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,205:utils:itr: 217, num_batch: 217, last loss: 1.250000, smooth_loss: 1.172401
[DEBUG]2020-06-19 09:25:14,259:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,294:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,262:utils:itr: 217, num_batch: 217, last loss: 1.078125, smooth_loss: 1.184535
[DEBUG]2020-06-19 09:25:14,297:utils:itr: 234, num_batch: 234, last loss: 1.226562, smooth_loss: 1.191851
[DEBUG]2020-06-19 09:25:14,336:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,338:utils:itr: 219, num_batch: 219, last loss: 1.125000, smooth_loss: 1.149118
[DEBUG]2020-06-19 09:25:14,410:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,435:utils:loss_avg: 1.16326, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.50444, mc_score:-0.00015
[DEBUG]2020-06-19 09:25:14,412:utils:itr: 229, num_batch: 229, last loss: 1.164062, smooth_loss: 1.168384
[DEBUG]2020-06-19 09:25:14,495:utils:loss_avg: 1.16399, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.47278, mc_score:-0.03182
[DEBUG]2020-06-19 09:25:14,531:utils:loss_avg: 1.17188, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.49269, mc_score:-0.01428
[DEBUG]2020-06-19 09:25:14,567:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,571:utils:loss_avg: 1.15680, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.48640, mc_score:-0.02774
[DEBUG]2020-06-19 09:25:14,574:utils:itr: 212, num_batch: 212, last loss: 1.164062, smooth_loss: 1.163466
[DEBUG]2020-06-19 09:25:14,645:utils:loss_avg: 1.19237, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.47664, mc_score:-0.01745
[DEBUG]2020-06-19 09:25:14,791:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,811:utils:loss_avg: 1.14809, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.48713, mc_score:0.00096
[DEBUG]2020-06-19 09:25:14,809:utils:itr: 229, num_batch: 229, last loss: 1.257812, smooth_loss: 1.147823
[DEBUG]2020-06-19 09:25:14,919:utils:on_backward_begin lr: 4e-05
[DEBUG]2020-06-19 09:25:14,946:utils:itr: 215, num_batch: 215, last loss: 1.218750, smooth_loss: 1.201453
[DEBUG]2020-06-19 09:25:15,068:utils:loss_avg: 1.15496, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.48563, mc_score:-0.00238
[DEBUG]2020-06-19 09:25:15,211:utils:loss_avg: 1.16341, lr_pg0:4e-05, lr_pg1: 4e-05final_score:0.49109, mc_score:-0.01159
[DEBUG]2020-06-19 09:25:16,268:utils:on_backward_begin lr: 4e-05
```
