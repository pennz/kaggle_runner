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
