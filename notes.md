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
## test result
For CPU, if use xla, it still report Grad None
    self.device = xm.xla_device(devkind='CPU')

For TPU, it will freeze when trying to get Grad

```
[DEBUG]2020-06-20 05:17:59,092:utils:loss_avg: 1.11405, lr_pg0:8.810488730080142e-07, lr_pg1: 8.810488730080142e-07final_score:0.48852, mc_score:0.03426
[DEBUG]2020-06-20 05:17:59,459:utils:grad info pg0: norm std(0.006225) mean(0.016256)
[DEBUG]2020-06-20 05:17:59,462:utils:grad info pg1: norm std(0.000734) mean(0.001428)
[DEBUG]2020-06-20 05:17:59,769:utils:on_backward_begin lr: 9.120108393559097e-07
[DEBUG]2020-06-20 05:17:59,771:utils:itr: 64, num_batch: 64, last loss: 1.044795, smooth_loss: 1.086013

[DEBUG]2020-06-20 05:17:59,780:utils:loss_avg: 1.11299, lr_pg0:9.120108393559097e-07, lr_pg1: 9.120108393559097e-07final_score:0.48463, mc_score:0.01871
[DEBUG]2020-06-20 05:18:00,147:utils:grad info pg0: norm std(0.008121) mean(0.019129)
[DEBUG]2020-06-20 05:18:00,151:utils:grad info pg1: norm std(0.000909) mean(0.001658)
[DEBUG]2020-06-20 05:18:00,452:utils:on_backward_begin lr: 9.440608762859234e-07
[DEBUG]2020-06-20 05:18:00,453:utils:itr: 65, num_batch: 65, last loss: 0.961200, smooth_loss: 1.082623
[DEBUG]2020-06-20 05:18:00,462:utils:loss_avg: 1.11069, lr_pg0:9.440608762859234e-07, lr_pg1: 9.440608762859234e-07final_score:0.48789, mc_score:0.01841
[DEBUG]2020-06-20 05:18:00,830:utils:grad info pg0: norm std(0.045156) mean(0.047097)
[DEBUG]2020-06-20 05:18:00,834:utils:grad info pg1: norm std(0.001272) mean(0.002412)
[DEBUG]2020-06-20 05:18:01,141:utils:on_backward_begin lr: 9.772372209558109e-07
[DEBUG]2020-06-20 05:18:01,143:utils:itr: 66, num_batch: 66, last loss: 0.894677, smooth_loss: 1.077555
[DEBUG]2020-06-20 05:18:01,152:utils:loss_avg: 1.10746, lr_pg0:9.772372209558109e-07, lr_pg1: 9.772372209558109e-07final_score:0.48752, mc_score:0.01298
[DEBUG]2020-06-20 05:18:01,519:utils:grad info pg0: norm std(0.004740) mean(0.014121)
[DEBUG]2020-06-20 05:18:01,522:utils:grad info pg1: norm std(0.000732) mean(0.001339)
[DEBUG]2020-06-20 05:18:01,834:utils:on_backward_begin lr: 1.0115794542598987e-06
[DEBUG]2020-06-20 05:18:01,836:utils:itr: 67, num_batch: 67, last loss: 1.236936, smooth_loss: 1.081823
[DEBUG]2020-06-20 05:18:01,849:utils:loss_avg: 1.10937, lr_pg0:1.0115794542598987e-06, lr_pg1: 1.0115794542598987e-06final_score:0.48646, mc_score:0.01334
[DEBUG]2020-06-20 05:18:02,216:utils:grad info pg0: norm std(0.014892) mean(0.045829)
[DEBUG]2020-06-20 05:18:02,220:utils:grad info pg1: norm std(0.002039) mean(0.003976)
[DEBUG]2020-06-20 05:18:02,565:utils:on_backward_begin lr: 1.0471285480508998e-06
[DEBUG]2020-06-20 05:18:02,567:utils:itr: 68, num_batch: 68, last loss: 1.214348, smooth_loss: 1.085348
[DEBUG]2020-06-20 05:18:02,578:utils:loss_avg: 1.11089, lr_pg0:1.0471285480508998e-06, lr_pg1: 1.0471285480508998e-06final_score:0.48536, mc_score:0.00680
[DEBUG]2020-06-20 05:18:02,945:utils:grad info pg0: norm std(0.003312) mean(0.011740)
[DEBUG]2020-06-20 05:18:02,949:utils:grad info pg1: norm std(0.000496) mean(0.000962)
[DEBUG]2020-06-20 05:18:03,256:utils:on_backward_begin lr: 1.0839269140212033e-06
[DEBUG]2020-06-20 05:18:03,259:utils:itr: 69, num_batch: 69, last loss: 1.228556, smooth_loss: 1.089132
[DEBUG]2020-06-20 05:18:03,268:utils:loss_avg: 1.11257, lr_pg0:1.0839269140212033e-06, lr_pg1: 1.0839269140212033e-06final_score:0.48444, mc_score:0.00040
[DEBUG]2020-06-20 05:18:03,642:utils:grad info pg0: norm std(0.015379) mean(0.050566)
[DEBUG]2020-06-20 05:18:03,645:utils:grad info pg1: norm std(0.002332) mean(0.004395)
[DEBUG]2020-06-20 05:18:03,956:utils:on_backward_begin lr: 1.1220184543019633e-06
[DEBUG]2020-06-20 05:18:03,957:utils:itr: 70, num_batch: 70, last loss: 0.865338, smooth_loss: 1.083256
[DEBUG]2020-06-20 05:18:03,967:utils:loss_avg: 1.10909, lr_pg0:1.1220184543019633e-06, lr_pg1: 1.1220184543019633e-06final_score:0.48403, mc_score:0.00420
[DEBUG]2020-06-20 05:18:04,335:utils:grad info pg0: norm std(0.012972) mean(0.027202)
[DEBUG]2020-06-20 05:18:04,339:utils:grad info pg1: norm std(0.001284) mean(0.002483)
[DEBUG]2020-06-20 05:18:04,639:utils:on_backward_begin lr: 1.1614486138403427e-06
[DEBUG]2020-06-20 05:18:04,641:utils:itr: 71, num_batch: 71, last loss: 0.918486, smooth_loss: 1.078957
[DEBUG]2020-06-20 05:18:04,651:utils:loss_avg: 1.10644, lr_pg0:1.1614486138403427e-06, lr_pg1: 1.1614486138403427e-06final_score:0.48783, mc_score:0.00637
[DEBUG]2020-06-20 05:18:05,019:utils:grad info pg0: norm std(0.009374) mean(0.022937)
[DEBUG]2020-06-20 05:18:05,023:utils:grad info pg1: norm std(0.001052) mean(0.002017)
[DEBUG]2020-06-20 05:18:05,328:utils:on_backward_begin lr: 1.2022644346174128e-06
[DEBUG]2020-06-20 05:18:05,329:utils:itr: 72, num_batch: 72, last loss: 0.976821, smooth_loss: 1.076308
[DEBUG]2020-06-20 05:18:05,340:utils:loss_avg: 1.10466, lr_pg0:1.2022644346174128e-06, lr_pg1: 1.2022644346174128e-06final_score:0.48413, mc_score:-0.00031
[DEBUG]2020-06-20 05:18:05,708:utils:grad info pg0: norm std(0.004530) mean(0.013096)
[DEBUG]2020-06-20 05:18:05,711:utils:grad info pg1: norm std(0.000611) mean(0.001148)
[DEBUG]2020-06-20 05:18:06,026:utils:on_backward_begin lr: 1.244514611771385e-06
[DEBUG]2020-06-20 05:18:06,028:utils:itr: 73, num_batch: 73, last loss: 1.186439, smooth_loss: 1.079148
[DEBUG]2020-06-20 05:18:06,037:utils:loss_avg: 1.10577, lr_pg0:1.244514611771385e-06, lr_pg1: 1.244514611771385e-06final_score:0.48452, mc_score:0.00000
[DEBUG]2020-06-20 05:18:06,405:utils:grad info pg0: norm std(0.018097) mean(0.034213)
[DEBUG]2020-06-20 05:18:06,408:utils:grad info pg1: norm std(0.001377) mean(0.002549)
[DEBUG]2020-06-20 05:18:06,708:utils:on_backward_begin lr: 1.288249551693134e-06
[DEBUG]2020-06-20 05:18:06,710:utils:itr: 74, num_batch: 74, last loss: 1.071715, smooth_loss: 1.078957
[DEBUG]2020-06-20 05:18:06,719:utils:loss_avg: 1.10532, lr_pg0:1.288249551693134e-06, lr_pg1: 1.288249551693134e-06final_score:0.48502, mc_score:0.00000
[DEBUG]2020-06-20 05:18:07,087:utils:grad info pg0: norm std(0.073270) mean(0.049054)
[DEBUG]2020-06-20 05:18:07,090:utils:grad info pg1: norm std(0.000879) mean(0.001807)
[DEBUG]2020-06-20 05:18:07,390:utils:on_backward_begin lr: 1.333521432163324e-06
[DEBUG]2020-06-20 05:18:07,392:utils:itr: 75, num_batch: 75, last loss: 0.971945, smooth_loss: 1.076230
[DEBUG]2020-06-20 05:18:07,401:utils:loss_avg: 1.10356, lr_pg0:1.333521432163324e-06, lr_pg1: 1.333521432163324e-06final_score:0.48457, mc_score:-0.00414
[DEBUG]2020-06-20 05:18:07,769:utils:grad info pg0: norm std(0.020636) mean(0.063667)
[DEBUG]2020-06-20 05:18:07,772:utils:grad info pg1: norm std(0.002434) mean(0.004426)
[DEBUG]2020-06-20 05:18:08,080:utils:on_backward_begin lr: 1.3803842646028848e-06
[DEBUG]2020-06-20 05:18:08,082:utils:itr: 76, num_batch: 76, last loss: 0.882366, smooth_loss: 1.071315
[DEBUG]2020-06-20 05:18:08,094:utils:loss_avg: 1.10069, lr_pg0:1.3803842646028848e-06, lr_pg1: 1.3803842646028848e-06final_score:0.48720, mc_score:-0.00022
[DEBUG]2020-06-20 05:18:08,461:utils:grad info pg0: norm std(0.008227) mean(0.026799)
[DEBUG]2020-06-20 05:18:08,465:utils:grad info pg1: norm std(0.001107) mean(0.002188)
[DEBUG]2020-06-20 05:18:08,776:utils:on_backward_begin lr: 1.428893958511103e-06
[DEBUG]2020-06-20 05:18:08,777:utils:itr: 77, num_batch: 77, last loss: 1.150728, smooth_loss: 1.073317
[DEBUG]2020-06-20 05:18:08,787:utils:loss_avg: 1.10133, lr_pg0:1.428893958511103e-06, lr_pg1: 1.428893958511103e-06final_score:0.48520, mc_score:-0.00403
[DEBUG]2020-06-20 05:18:09,156:utils:grad info pg0: norm std(0.006526) mean(0.022343)
[DEBUG]2020-06-20 05:18:09,159:utils:grad info pg1: norm std(0.001016) mean(0.001956)
[DEBUG]2020-06-20 05:18:09,460:utils:on_backward_begin lr: 1.4791083881682077e-06
[DEBUG]2020-06-20 05:18:09,461:utils:itr: 78, num_batch: 78, last loss: 0.967090, smooth_loss: 1.070653
[DEBUG]2020-06-20 05:18:09,470:utils:loss_avg: 1.09963, lr_pg0:1.4791083881682077e-06, lr_pg1: 1.4791083881682077e-06final_score:0.48659, mc_score:-0.00562
[DEBUG]2020-06-20 05:18:09,843:utils:grad info pg0: norm std(0.035264) mean(0.037154)
[DEBUG]2020-06-20 05:18:09,847:utils:grad info pg1: norm std(0.000883) mean(0.001778)
[DEBUG]2020-06-20 05:18:10,158:utils:on_backward_begin lr: 1.5310874616820303e-06
[DEBUG]2020-06-20 05:18:10,161:utils:itr: 79, num_batch: 79, last loss: 1.003126, smooth_loss: 1.068967
[DEBUG]2020-06-20 05:18:10,172:utils:loss_avg: 1.09842, lr_pg0:1.5310874616820303e-06, lr_pg1: 1.5310874616820303e-06final_score:0.48739, mc_score:-0.00728
[DEBUG]2020-06-20 05:18:10,541:utils:grad info pg0: norm std(0.006061) mean(0.015658)
[DEBUG]2020-06-20 05:18:10,545:utils:grad info pg1: norm std(0.000683) mean(0.001301)
[DEBUG]2020-06-20 05:18:10,850:utils:on_backward_begin lr: 1.5848931924611137e-06
[DEBUG]2020-06-20 05:18:10,852:utils:itr: 80, num_batch: 80, last loss: 0.983854, smooth_loss: 1.066854

[DEBUG]2020-06-20 05:18:10,863:utils:loss_avg: 1.09701, lr_pg0:1.5848931924611137e-06, lr_pg1: 1.5848931924611137e-06final_score:0.48569, mc_score:-0.00731
[DEBUG]2020-06-20 05:18:11,236:utils:grad info pg0: norm std(0.011225) mean(0.036313)
[DEBUG]2020-06-20 05:18:11,240:utils:grad info pg1: norm std(0.001475) mean(0.002960)
[DEBUG]2020-06-20 05:18:11,540:utils:on_backward_begin lr: 1.6405897731995394e-06
[DEBUG]2020-06-20 05:18:11,542:utils:itr: 81, num_batch: 81, last loss: 1.259587, smooth_loss: 1.071617
[DEBUG]2020-06-20 05:18:11,551:utils:loss_avg: 1.09899, lr_pg0:1.6405897731995394e-06, lr_pg1: 1.6405897731995394e-06final_score:0.48434, mc_score:-0.01254
[DEBUG]2020-06-20 05:18:11,920:utils:grad info pg0: norm std(0.029734) mean(0.033607)
[DEBUG]2020-06-20 05:18:11,923:utils:grad info pg1: norm std(0.000993) mean(0.001952)
[DEBUG]2020-06-20 05:18:12,224:utils:on_backward_begin lr: 1.698243652461744e-06
[DEBUG]2020-06-20 05:18:12,226:utils:itr: 82, num_batch: 82, last loss: 1.020243, smooth_loss: 1.070354
[DEBUG]2020-06-20 05:18:12,237:utils:loss_avg: 1.09804, lr_pg0:1.698243652461744e-06, lr_pg1: 1.698243652461744e-06final_score:0.48612, mc_score:-0.01059
[DEBUG]2020-06-20 05:18:12,612:utils:grad info pg0: norm std(0.004206) mean(0.012953)
[DEBUG]2020-06-20 05:18:12,616:utils:grad info pg1: norm std(0.000549) mean(0.001065)
[DEBUG]2020-06-20 05:18:12,928:utils:on_backward_begin lr: 1.7579236139586924e-06
[DEBUG]2020-06-20 05:18:12,930:utils:itr: 83, num_batch: 83, last loss: 0.848575, smooth_loss: 1.064923
[DEBUG]2020-06-20 05:18:12,943:utils:loss_avg: 1.09507, lr_pg0:1.7579236139586924e-06, lr_pg1: 1.7579236139586924e-06final_score:0.49021, mc_score:-0.00864
[DEBUG]2020-06-20 05:18:13,314:utils:grad info pg0: norm std(0.039244) mean(0.063938)
[DEBUG]2020-06-20 05:18:13,318:utils:grad info pg1: norm std(0.002087) mean(0.004245)
[DEBUG]2020-06-20 05:18:13,622:utils:on_backward_begin lr: 1.8197008586099832e-06
[DEBUG]2020-06-20 05:18:13,623:utils:itr: 84, num_batch: 84, last loss: 1.090404, smooth_loss: 1.065544
[DEBUG]2020-06-20 05:18:13,633:utils:loss_avg: 1.09502, lr_pg0:1.8197008586099832e-06, lr_pg1: 1.8197008586099832e-06final_score:0.48663, mc_score:-0.00868
[DEBUG]2020-06-20 05:18:14,006:utils:grad info pg0: norm std(0.039520) mean(0.039945)
[DEBUG]2020-06-20 05:18:14,009:utils:grad info pg1: norm std(0.001273) mean(0.002360)
[DEBUG]2020-06-20 05:18:14,308:utils:on_backward_begin lr: 1.8836490894898005e-06
[DEBUG]2020-06-20 05:18:14,309:utils:itr: 85, num_batch: 85, last loss: 1.024877, smooth_loss: 1.064557
[DEBUG]2020-06-20 05:18:14,320:utils:loss_avg: 1.09420, lr_pg0:1.8836490894898005e-06, lr_pg1: 1.8836490894898005e-06final_score:0.48780, mc_score:-0.01359
[DEBUG]2020-06-20 05:18:14,688:utils:grad info pg0: norm std(0.076903) mean(0.068190)
[DEBUG]2020-06-20 05:18:14,691:utils:grad info pg1: norm std(0.001895) mean(0.003370)
[DEBUG]2020-06-20 05:18:14,990:utils:on_backward_begin lr: 1.9498445997580454e-06
[DEBUG]2020-06-20 05:18:14,992:utils:itr: 86, num_batch: 86, last loss: 1.102917, smooth_loss: 1.065484
[DEBUG]2020-06-20 05:18:15,003:utils:loss_avg: 1.09430, lr_pg0:1.9498445997580454e-06, lr_pg1: 1.9498445997580454e-06final_score:0.48920, mc_score:-0.00996
[DEBUG]2020-06-20 05:18:15,376:utils:grad info pg0: norm std(0.016545) mean(0.017940)
[DEBUG]2020-06-20 05:18:15,380:utils:grad info pg1: norm std(0.000580) mean(0.001100)
[DEBUG]2020-06-20 05:18:15,681:utils:on_backward_begin lr: 2.018366363681561e-06
[DEBUG]2020-06-20 05:18:15,682:utils:itr: 87, num_batch: 87, last loss: 1.210692, smooth_loss: 1.068979
[DEBUG]2020-06-20 05:18:15,691:utils:loss_avg: 1.09563, lr_pg0:2.018366363681561e-06, lr_pg1: 2.018366363681561e-06final_score:0.48612, mc_score:-0.01974
[DEBUG]2020-06-20 05:18:16,065:utils:grad info pg0: norm std(0.010685) mean(0.028040)
[DEBUG]2020-06-20 05:18:16,068:utils:grad info pg1: norm std(0.001307) mean(0.002437)
[DEBUG]2020-06-20 05:18:16,368:utils:on_backward_begin lr: 2.0892961308540394e-06
[DEBUG]2020-06-20 05:18:16,370:utils:itr: 88, num_batch: 88, last loss: 1.041082, smooth_loss: 1.068310
[DEBUG]2020-06-20 05:18:16,380:utils:loss_avg: 1.09501, lr_pg0:2.0892961308540394e-06, lr_pg1: 2.0892961308540394e-06final_score:0.48758, mc_score:-0.01460
[DEBUG]2020-06-20 05:18:16,748:utils:grad info pg0: norm std(0.012270) mean(0.046251)
[DEBUG]2020-06-20 05:18:16,752:utils:grad info pg1: norm std(0.001731) mean(0.003319)
[DEBUG]2020-06-20 05:18:17,068:utils:on_backward_begin lr: 2.1627185237270203e-06
[DEBUG]2020-06-20 05:18:17,069:utils:itr: 89, num_batch: 89, last loss: 1.065762, smooth_loss: 1.068249
[DEBUG]2020-06-20 05:18:17,080:utils:loss_avg: 1.09469, lr_pg0:2.1627185237270203e-06, lr_pg1: 2.1627185237270203e-06final_score:0.48670, mc_score:-0.01443
[DEBUG]2020-06-20 05:18:17,453:utils:grad info pg0: norm std(0.013809) mean(0.038481)
[DEBUG]2020-06-20 05:18:17,457:utils:grad info pg1: norm std(0.001498) mean(0.003020)
[DEBUG]2020-06-20 05:18:17,755:utils:on_backward_begin lr: 2.23872113856834e-06
[DEBUG]2020-06-20 05:18:17,756:utils:itr: 90, num_batch: 90, last loss: 0.959231, smooth_loss: 1.065657
[DEBUG]2020-06-20 05:18:17,766:utils:loss_avg: 1.09320, lr_pg0:2.23872113856834e-06, lr_pg1: 2.23872113856834e-06final_score:0.48643, mc_score:-0.01744
[DEBUG]2020-06-20 05:18:18,135:utils:grad info pg0: norm std(0.098228) mean(0.067119)
[DEBUG]2020-06-20 05:18:18,139:utils:grad info pg1: norm std(0.001206) mean(0.002397)
[DEBUG]2020-06-20 05:18:18,456:utils:on_backward_begin lr: 2.317394649968479e-06
[DEBUG]2020-06-20 05:18:18,457:utils:itr: 91, num_batch: 91, last loss: 0.805971, smooth_loss: 1.059504
[DEBUG]2020-06-20 05:18:18,467:utils:loss_avg: 1.09008, lr_pg0:2.317394649968479e-06, lr_pg1: 2.317394649968479e-06final_score:0.48935, mc_score:-0.01101
[DEBUG]2020-06-20 05:18:18,839:utils:grad info pg0: norm std(0.013201) mean(0.041009)
[DEBUG]2020-06-20 05:18:18,843:utils:grad info pg1: norm std(0.001485) mean(0.002993)
[DEBUG]2020-06-20 05:18:19,145:utils:on_backward_begin lr: 2.398832919019491e-06
[DEBUG]2020-06-20 05:18:19,147:utils:itr: 92, num_batch: 92, last loss: 0.885852, smooth_loss: 1.055405
[DEBUG]2020-06-20 05:18:19,157:utils:loss_avg: 1.08788, lr_pg0:2.398832919019491e-06, lr_pg1: 2.398832919019491e-06final_score:0.49204, mc_score:-0.00634
[DEBUG]2020-06-20 05:18:19,531:utils:grad info pg0: norm std(0.006949) mean(0.021473)
[DEBUG]2020-06-20 05:18:19,534:utils:grad info pg1: norm std(0.000949) mean(0.001853)
[DEBUG]2020-06-20 05:18:19,836:utils:on_backward_begin lr: 2.4831331052955706e-06
[DEBUG]2020-06-20 05:18:19,837:utils:itr: 93, num_batch: 93, last loss: 1.037075, smooth_loss: 1.054974
[DEBUG]2020-06-20 05:18:19,849:utils:loss_avg: 1.08734, lr_pg0:2.4831331052955706e-06, lr_pg1: 2.4831331052955706e-06final_score:0.49201, mc_score:-0.00632
[DEBUG]2020-06-20 05:18:20,217:utils:grad info pg0: norm std(0.011620) mean(0.030598)
[DEBUG]2020-06-20 05:18:20,221:utils:grad info pg1: norm std(0.001220) mean(0.002434)
[DEBUG]2020-06-20 05:18:20,540:utils:on_backward_begin lr: 2.5703957827688634e-06
[DEBUG]2020-06-20 05:18:20,542:utils:itr: 94, num_batch: 94, last loss: 1.079897, smooth_loss: 1.055558
[DEBUG]2020-06-20 05:18:20,551:utils:loss_avg: 1.08726, lr_pg0:2.5703957827688634e-06, lr_pg1: 2.5703957827688634e-06final_score:0.49236, mc_score:-0.00781
[DEBUG]2020-06-20 05:18:20,918:utils:grad info pg0: norm std(0.013618) mean(0.047994)
[DEBUG]2020-06-20 05:18:20,921:utils:grad info pg1: norm std(0.002301) mean(0.004160)
[DEBUG]2020-06-20 05:18:21,223:utils:on_backward_begin lr: 2.6607250597988094e-06
[DEBUG]2020-06-20 05:18:21,224:utils:itr: 95, num_batch: 95, last loss: 1.183720, smooth_loss: 1.058552
[DEBUG]2020-06-20 05:18:21,235:utils:loss_avg: 1.08827, lr_pg0:2.6607250597988094e-06, lr_pg1: 2.6607250597988094e-06final_score:0.49245, mc_score:-0.00781
[DEBUG]2020-06-20 05:18:21,603:utils:grad info pg0: norm std(0.004136) mean(0.012371)
[DEBUG]2020-06-20 05:18:21,606:utils:grad info pg1: norm std(0.000564) mean(0.001069)
[DEBUG]2020-06-20 05:18:21,904:utils:on_backward_begin lr: 2.754228703338166e-06
[DEBUG]2020-06-20 05:18:21,905:utils:itr: 96, num_batch: 96, last loss: 0.913632, smooth_loss: 1.055178

[DEBUG]2020-06-20 05:18:21,915:utils:loss_avg: 1.08647, lr_pg0:2.754228703338166e-06, lr_pg1: 2.754228703338166e-06final_score:0.49407, mc_score:-0.00348
[DEBUG]2020-06-20 05:18:22,283:utils:grad info pg0: norm std(0.011153) mean(0.032725)
[DEBUG]2020-06-20 05:18:22,286:utils:grad info pg1: norm std(0.001390) mean(0.002692)
[DEBUG]2020-06-20 05:18:22,585:utils:on_backward_begin lr: 2.8510182675039094e-06
[DEBUG]2020-06-20 05:18:22,586:utils:itr: 97, num_batch: 97, last loss: 0.855278, smooth_loss: 1.050539
[DEBUG]2020-06-20 05:18:22,595:utils:loss_avg: 1.08411, lr_pg0:2.8510182675039094e-06, lr_pg1: 2.8510182675039094e-06final_score:0.49651, mc_score:0.00058
[DEBUG]2020-06-20 05:18:22,963:utils:grad info pg0: norm std(0.020514) mean(0.040969)
[DEBUG]2020-06-20 05:18:22,966:utils:grad info pg1: norm std(0.002207) mean(0.003948)
[DEBUG]2020-06-20 05:18:23,268:utils:on_backward_begin lr: 2.9512092266663854e-06
[DEBUG]2020-06-20 05:18:23,270:utils:itr: 98, num_batch: 98, last loss: 1.034990, smooth_loss: 1.050180
[DEBUG]2020-06-20 05:18:23,280:utils:loss_avg: 1.08361, lr_pg0:2.9512092266663854e-06, lr_pg1: 2.9512092266663854e-06final_score:0.49508, mc_score:-0.00371
[DEBUG]2020-06-20 05:18:23,653:utils:grad info pg0: norm std(0.008644) mean(0.023338)
[DEBUG]2020-06-20 05:18:23,657:utils:grad info pg1: norm std(0.001202) mean(0.002184)
[DEBUG]2020-06-20 05:18:23,970:utils:on_backward_begin lr: 3.054921113215513e-06
[DEBUG]2020-06-20 05:18:23,971:utils:itr: 99, num_batch: 99, last loss: 0.963697, smooth_loss: 1.048186
[DEBUG]2020-06-20 05:18:23,983:utils:loss_avg: 1.08241, lr_pg0:3.054921113215513e-06, lr_pg1: 3.054921113215513e-06final_score:0.49341, mc_score:-0.01078
[DEBUG]2020-06-20 05:18:24,356:utils:grad info pg0: norm std(0.030635) mean(0.035870)
[DEBUG]2020-06-20 05:18:24,359:utils:grad info pg1: norm std(0.001505) mean(0.002606)
[DEBUG]2020-06-20 05:18:24,666:utils:on_backward_begin lr: 3.1622776601683796e-06
[DEBUG]2020-06-20 05:18:24,668:utils:itr: 100, num_batch: 100, last loss: 0.799141, smooth_loss: 1.042461
[DEBUG]2020-06-20 05:18:24,678:utils:loss_avg: 1.07961, lr_pg0:3.1622776601683796e-06, lr_pg1: 3.1622776601683796e-06final_score:0.49880, mc_score:0.00112
[DEBUG]2020-06-20 05:18:25,045:utils:grad info pg0: norm std(0.005910) mean(0.017407)
[DEBUG]2020-06-20 05:18:25,049:utils:grad info pg1: norm std(0.000752) mean(0.001457)
[DEBUG]2020-06-20 05:18:25,349:utils:on_backward_begin lr: 3.273406948788382e-06
[DEBUG]2020-06-20 05:18:25,351:utils:itr: 101, num_batch: 101, last loss: 0.952661, smooth_loss: 1.040403
[DEBUG]2020-06-20 05:18:25,360:utils:loss_avg: 1.07836, lr_pg0:3.273406948788382e-06, lr_pg1: 3.273406948788382e-06final_score:0.49871, mc_score:0.00378
[DEBUG]2020-06-20 05:18:25,728:utils:grad info pg0: norm std(0.010536) mean(0.027586)
[DEBUG]2020-06-20 05:18:25,731:utils:grad info pg1: norm std(0.001101) mean(0.002215)
[DEBUG]2020-06-20 05:18:26,049:utils:on_backward_begin lr: 3.388441561392026e-06
[DEBUG]2020-06-20 05:18:26,051:utils:itr: 102, num_batch: 102, last loss: 0.765022, smooth_loss: 1.034109
[DEBUG]2020-06-20 05:18:26,062:utils:loss_avg: 1.07532, lr_pg0:3.388441561392026e-06, lr_pg1: 3.388441561392026e-06final_score:0.50476, mc_score:0.01243
[DEBUG]2020-06-20 05:18:26,430:utils:grad info pg0: norm std(0.012479) mean(0.030560)
[DEBUG]2020-06-20 05:18:26,433:utils:grad info pg1: norm std(0.001414) mean(0.002731)
[DEBUG]2020-06-20 05:18:26,731:utils:on_backward_begin lr: 3.5075187395256803e-06
[DEBUG]2020-06-20 05:18:26,732:utils:itr: 103, num_batch: 103, last loss: 0.785730, smooth_loss: 1.028449
[DEBUG]2020-06-20 05:18:26,742:utils:loss_avg: 1.07254, lr_pg0:3.5075187395256803e-06, lr_pg1: 3.5075187395256803e-06final_score:0.50972, mc_score:0.02075
[DEBUG]2020-06-20 05:18:27,108:utils:grad info pg0: norm std(0.065864) mean(0.061409)
[DEBUG]2020-06-20 05:18:27,111:utils:grad info pg1: norm std(0.001763) mean(0.003218)
[DEBUG]2020-06-20 05:18:27,413:utils:on_backward_begin lr: 3.630780547701014e-06
[DEBUG]2020-06-20 05:18:27,415:utils:itr: 104, num_batch: 104, last loss: 1.167730, smooth_loss: 1.031615
[DEBUG]2020-06-20 05:18:27,424:utils:loss_avg: 1.07344, lr_pg0:3.630780547701014e-06, lr_pg1: 3.630780547701014e-06final_score:0.50634, mc_score:0.01387
[DEBUG]2020-06-20 05:18:27,793:utils:grad info pg0: norm std(0.010001) mean(0.024564)
[DEBUG]2020-06-20 05:18:27,796:utils:grad info pg1: norm std(0.001120) mean(0.002130)
[DEBUG]2020-06-20 05:18:28,112:utils:on_backward_begin lr: 3.758374042884442e-06
[DEBUG]2020-06-20 05:18:28,113:utils:itr: 105, num_batch: 105, last loss: 1.096754, smooth_loss: 1.033091
[DEBUG]2020-06-20 05:18:28,124:utils:loss_avg: 1.07366, lr_pg0:3.758374042884442e-06, lr_pg1: 3.758374042884442e-06final_score:0.50645, mc_score:0.01571
[DEBUG]2020-06-20 05:18:28,492:utils:grad info pg0: norm std(0.007875) mean(0.016722)
[DEBUG]2020-06-20 05:18:28,496:utils:grad info pg1: norm std(0.000742) mean(0.001379)
[DEBUG]2020-06-20 05:18:28,797:utils:on_backward_begin lr: 3.890451449942807e-06
[DEBUG]2020-06-20 05:18:28,798:utils:itr: 106, num_batch: 106, last loss: 1.021798, smooth_loss: 1.032836
[DEBUG]2020-06-20 05:18:28,807:utils:loss_avg: 1.07318, lr_pg0:3.890451449942807e-06, lr_pg1: 3.890451449942807e-06final_score:0.50561, mc_score:0.01420
[DEBUG]2020-06-20 05:18:29,181:utils:grad info pg0: norm std(0.014350) mean(0.043318)
[DEBUG]2020-06-20 05:18:29,184:utils:grad info pg1: norm std(0.001771) mean(0.003487)
[DEBUG]2020-06-20 05:18:29,491:utils:on_backward_begin lr: 4.027170343254592e-06
[DEBUG]2020-06-20 05:18:29,493:utils:itr: 107, num_batch: 107, last loss: 0.911258, smooth_loss: 1.030095
[DEBUG]2020-06-20 05:18:29,503:utils:loss_avg: 1.07168, lr_pg0:4.027170343254592e-06, lr_pg1: 4.027170343254592e-06final_score:0.50784, mc_score:0.01835
[DEBUG]2020-06-20 05:18:29,875:utils:grad info pg0: norm std(0.006971) mean(0.023908)
[DEBUG]2020-06-20 05:18:29,879:utils:grad info pg1: norm std(0.001065) mean(0.002034)
[DEBUG]2020-06-20 05:18:30,187:utils:on_backward_begin lr: 4.168693834703355e-06
[DEBUG]2020-06-20 05:18:30,189:utils:itr: 108, num_batch: 108, last loss: 0.952855, smooth_loss: 1.028358
[DEBUG]2020-06-20 05:18:30,199:utils:loss_avg: 1.07059, lr_pg0:4.168693834703355e-06, lr_pg1: 4.168693834703355e-06final_score:0.51027, mc_score:0.01848
[DEBUG]2020-06-20 05:18:30,567:utils:grad info pg0: norm std(0.003105) mean(0.008752)
[DEBUG]2020-06-20 05:18:30,570:utils:grad info pg1: norm std(0.000413) mean(0.000782)
[DEBUG]2020-06-20 05:18:30,870:utils:on_backward_begin lr: 4.315190768277653e-06
[DEBUG]2020-06-20 05:18:30,871:utils:itr: 109, num_batch: 109, last loss: 0.897537, smooth_loss: 1.025424
[DEBUG]2020-06-20 05:18:30,881:utils:loss_avg: 1.06902, lr_pg0:4.315190768277653e-06, lr_pg1: 4.315190768277653e-06final_score:0.51454, mc_score:0.02551
[DEBUG]2020-06-20 05:18:31,248:utils:grad info pg0: norm std(0.013819) mean(0.024426)
[DEBUG]2020-06-20 05:18:31,252:utils:grad info pg1: norm std(0.000927) mean(0.001790)
[DEBUG]2020-06-20 05:18:31,550:utils:on_backward_begin lr: 4.466835921509633e-06
[DEBUG]2020-06-20 05:18:31,551:utils:itr: 110, num_batch: 110, last loss: 0.760915, smooth_loss: 1.019505
[DEBUG]2020-06-20 05:18:31,561:utils:loss_avg: 1.06624, lr_pg0:4.466835921509633e-06, lr_pg1: 4.466835921509633e-06final_score:0.51746, mc_score:0.03036
[DEBUG]2020-06-20 05:18:31,928:utils:grad info pg0: norm std(0.030277) mean(0.045063)
[DEBUG]2020-06-20 05:18:31,931:utils:grad info pg1: norm std(0.001480) mean(0.002784)
[DEBUG]2020-06-20 05:18:32,234:utils:on_backward_begin lr: 4.623810213992605e-06
[DEBUG]2020-06-20 05:18:32,236:utils:itr: 111, num_batch: 111, last loss: 1.084978, smooth_loss: 1.020966
[DEBUG]2020-06-20 05:18:32,246:utils:loss_avg: 1.06641, lr_pg0:4.623810213992605e-06, lr_pg1: 4.623810213992605e-06final_score:0.51583, mc_score:0.02470
[DEBUG]2020-06-20 05:18:32,619:utils:grad info pg0: norm std(0.038810) mean(0.068839)
[DEBUG]2020-06-20 05:18:32,622:utils:grad info pg1: norm std(0.004675) mean(0.006567)
[DEBUG]2020-06-20 05:18:32,924:utils:on_backward_begin lr: 4.786300923226385e-06
[DEBUG]2020-06-20 05:18:32,925:utils:itr: 112, num_batch: 112, last loss: 1.209248, smooth_loss: 1.025160

[DEBUG]2020-06-20 05:18:32,935:utils:loss_avg: 1.06767, lr_pg0:4.786300923226385e-06, lr_pg1: 4.786300923226385e-06final_score:0.51297, mc_score:0.01970
[DEBUG]2020-06-20 05:18:33,301:utils:grad info pg0: norm std(0.018005) mean(0.025912)
[DEBUG]2020-06-20 05:18:33,305:utils:grad info pg1: norm std(0.000871) mean(0.001725)
[DEBUG]2020-06-20 05:18:33,605:utils:on_backward_begin lr: 4.954501908047901e-06
[DEBUG]2020-06-20 05:18:33,606:utils:itr: 113, num_batch: 113, last loss: 1.088954, smooth_loss: 1.026577
[DEBUG]2020-06-20 05:18:33,617:utils:loss_avg: 1.06786, lr_pg0:4.954501908047901e-06, lr_pg1: 4.954501908047901e-06final_score:0.51226, mc_score:0.01684
[DEBUG]2020-06-20 05:18:33,984:utils:grad info pg0: norm std(0.075649) mean(0.052738)
[DEBUG]2020-06-20 05:18:33,987:utils:grad info pg1: norm std(0.001026) mean(0.002036)
[DEBUG]2020-06-20 05:18:34,283:utils:on_backward_begin lr: 5.128613839913646e-06
[DEBUG]2020-06-20 05:18:34,285:utils:itr: 114, num_batch: 114, last loss: 0.942617, smooth_loss: 1.024716
[DEBUG]2020-06-20 05:18:34,294:utils:loss_avg: 1.06677, lr_pg0:5.128613839913646e-06, lr_pg1: 5.128613839913646e-06final_score:0.51361, mc_score:0.01692
[DEBUG]2020-06-20 05:18:34,662:utils:grad info pg0: norm std(0.006090) mean(0.016332)
[DEBUG]2020-06-20 05:18:34,665:utils:grad info pg1: norm std(0.000778) mean(0.001451)
[DEBUG]2020-06-20 05:18:34,984:utils:on_backward_begin lr: 5.308844442309882e-06
[DEBUG]2020-06-20 05:18:34,985:utils:itr: 115, num_batch: 115, last loss: 0.986301, smooth_loss: 1.023866
[DEBUG]2020-06-20 05:18:34,994:utils:loss_avg: 1.06608, lr_pg0:5.308844442309882e-06, lr_pg1: 5.308844442309882e-06final_score:0.51295, mc_score:0.01391
[DEBUG]2020-06-20 05:18:35,368:utils:grad info pg0: norm std(0.004861) mean(0.013325)
[DEBUG]2020-06-20 05:18:35,373:utils:grad info pg1: norm std(0.000642) mean(0.001192)
[DEBUG]2020-06-20 05:18:35,687:utils:on_backward_begin lr: 5.495408738576244e-06
[DEBUG]2020-06-20 05:18:35,688:utils:itr: 116, num_batch: 116, last loss: 1.100381, smooth_loss: 1.025555
[DEBUG]2020-06-20 05:18:35,699:utils:loss_avg: 1.06637, lr_pg0:5.495408738576244e-06, lr_pg1: 5.495408738576244e-06final_score:0.51010, mc_score:0.01187
[DEBUG]2020-06-20 05:18:36,066:utils:grad info pg0: norm std(0.013337) mean(0.038984)
[DEBUG]2020-06-20 05:18:36,069:utils:grad info pg1: norm std(0.001719) mean(0.003218)
[DEBUG]2020-06-20 05:18:36,368:utils:on_backward_begin lr: 5.688529308438414e-06
[DEBUG]2020-06-20 05:18:36,369:utils:itr: 117, num_batch: 117, last loss: 0.928204, smooth_loss: 1.023410
[DEBUG]2020-06-20 05:18:36,379:utils:loss_avg: 1.06520, lr_pg0:5.688529308438414e-06, lr_pg1: 5.688529308438414e-06final_score:0.51131, mc_score:0.01352
[DEBUG]2020-06-20 05:18:36,747:utils:grad info pg0: norm std(0.012756) mean(0.040475)
[DEBUG]2020-06-20 05:18:36,750:utils:grad info pg1: norm std(0.001578) mean(0.003083)
[DEBUG]2020-06-20 05:18:37,048:utils:on_backward_begin lr: 5.888436553555888e-06
[DEBUG]2020-06-20 05:18:37,050:utils:itr: 118, num_batch: 118, last loss: 1.170437, smooth_loss: 1.026643
[DEBUG]2020-06-20 05:18:37,060:utils:loss_avg: 1.06608, lr_pg0:5.888436553555888e-06, lr_pg1: 5.888436553555888e-06final_score:0.50931, mc_score:0.01093
[DEBUG]2020-06-20 05:18:37,433:utils:grad info pg0: norm std(0.012353) mean(0.031940)
[DEBUG]2020-06-20 05:18:37,436:utils:grad info pg1: norm std(0.001334) mean(0.002599)
[DEBUG]2020-06-20 05:18:37,742:utils:on_backward_begin lr: 6.095368972401691e-06
[DEBUG]2020-06-20 05:18:37,743:utils:itr: 119, num_batch: 119, last loss: 0.772449, smooth_loss: 1.021065
[DEBUG]2020-06-20 05:18:37,755:utils:loss_avg: 1.06363, lr_pg0:6.095368972401691e-06, lr_pg1: 6.095368972401691e-06final_score:0.51265, mc_score:0.01647
[DEBUG]2020-06-20 05:18:38,124:utils:grad info pg0: norm std(0.013081) mean(0.032738)
[DEBUG]2020-06-20 05:18:38,128:utils:grad info pg1: norm std(0.001722) mean(0.003149)
[DEBUG]2020-06-20 05:18:38,427:utils:on_backward_begin lr: 6.309573444801932e-06
[DEBUG]2020-06-20 05:18:38,429:utils:itr: 120, num_batch: 120, last loss: 1.061813, smooth_loss: 1.021958
[DEBUG]2020-06-20 05:18:38,439:utils:loss_avg: 1.06362, lr_pg0:6.309573444801932e-06, lr_pg1: 6.309573444801932e-06final_score:0.51198, mc_score:0.01239
[DEBUG]2020-06-20 05:18:38,806:utils:grad info pg0: norm std(0.010821) mean(0.024853)
[DEBUG]2020-06-20 05:18:38,810:utils:grad info pg1: norm std(0.001334) mean(0.002477)
[DEBUG]2020-06-20 05:18:39,135:utils:on_backward_begin lr: 6.531305526474723e-06
[DEBUG]2020-06-20 05:18:39,137:utils:itr: 121, num_batch: 121, last loss: 0.939076, smooth_loss: 1.020146
[DEBUG]2020-06-20 05:18:39,148:utils:loss_avg: 1.06260, lr_pg0:6.531305526474723e-06, lr_pg1: 6.531305526474723e-06final_score:0.51104, mc_score:0.01513
[DEBUG]2020-06-20 05:18:39,521:utils:grad info pg0: norm std(0.011575) mean(0.036283)
[DEBUG]2020-06-20 05:18:39,525:utils:grad info pg1: norm std(0.002164) mean(0.003518)
[DEBUG]2020-06-20 05:18:39,823:utils:on_backward_begin lr: 6.760829753919817e-06
[DEBUG]2020-06-20 05:18:39,824:utils:itr: 122, num_batch: 122, last loss: 1.276448, smooth_loss: 1.025738
[DEBUG]2020-06-20 05:18:39,835:utils:loss_avg: 1.06434, lr_pg0:6.760829753919817e-06, lr_pg1: 6.760829753919817e-06final_score:0.50680, mc_score:0.00911
[DEBUG]2020-06-20 05:18:40,244:utils:grad info pg0: norm std(0.005570) mean(0.014793)
[DEBUG]2020-06-20 05:18:40,247:utils:grad info pg1: norm std(0.000629) mean(0.001210)
[DEBUG]2020-06-20 05:18:40,546:utils:on_backward_begin lr: 6.998419960022735e-06
[DEBUG]2020-06-20 05:18:40,547:utils:itr: 123, num_batch: 123, last loss: 0.951081, smooth_loss: 1.024112
[DEBUG]2020-06-20 05:18:40,557:utils:loss_avg: 1.06342, lr_pg0:6.998419960022735e-06, lr_pg1: 6.998419960022735e-06final_score:0.50700, mc_score:0.00970
[DEBUG]2020-06-20 05:18:40,930:utils:grad info pg0: norm std(0.024688) mean(0.028875)
[DEBUG]2020-06-20 05:18:40,934:utils:grad info pg1: norm std(0.000843) mean(0.001641)
[DEBUG]2020-06-20 05:18:41,234:utils:on_backward_begin lr: 7.244359600749901e-06
[DEBUG]2020-06-20 05:18:41,235:utils:itr: 124, num_batch: 124, last loss: 1.055055, smooth_loss: 1.024785
[DEBUG]2020-06-20 05:18:41,246:utils:loss_avg: 1.06336, lr_pg0:7.244359600749901e-06, lr_pg1: 7.244359600749901e-06final_score:0.50569, mc_score:0.00899
[DEBUG]2020-06-20 05:18:41,614:utils:grad info pg0: norm std(0.045640) mean(0.040656)
[DEBUG]2020-06-20 05:18:41,617:utils:grad info pg1: norm std(0.000876) mean(0.001791)
[DEBUG]2020-06-20 05:18:41,920:utils:on_backward_begin lr: 7.498942093324558e-06
[DEBUG]2020-06-20 05:18:41,921:utils:itr: 125, num_batch: 125, last loss: 0.854247, smooth_loss: 1.021084
[DEBUG]2020-06-20 05:18:41,933:utils:loss_avg: 1.06170, lr_pg0:7.498942093324558e-06, lr_pg1: 7.498942093324558e-06final_score:0.50669, mc_score:0.01133
[DEBUG]2020-06-20 05:18:42,307:utils:grad info pg0: norm std(0.004009) mean(0.012379)
[DEBUG]2020-06-20 05:18:42,311:utils:grad info pg1: norm std(0.000647) mean(0.001189)
[DEBUG]2020-06-20 05:18:42,621:utils:on_backward_begin lr: 7.762471166286918e-06
[DEBUG]2020-06-20 05:18:42,623:utils:itr: 126, num_batch: 126, last loss: 0.851377, smooth_loss: 1.017407
[DEBUG]2020-06-20 05:18:42,632:utils:loss_avg: 1.06004, lr_pg0:7.762471166286918e-06, lr_pg1: 7.762471166286918e-06final_score:0.50831, mc_score:0.01115
[DEBUG]2020-06-20 05:18:43,000:utils:grad info pg0: norm std(0.005928) mean(0.014068)
[DEBUG]2020-06-20 05:18:43,003:utils:grad info pg1: norm std(0.000717) mean(0.001312)
[DEBUG]2020-06-20 05:18:43,301:utils:on_backward_begin lr: 8.035261221856173e-06
[DEBUG]2020-06-20 05:18:43,303:utils:itr: 127, num_batch: 127, last loss: 0.979158, smooth_loss: 1.016580
[DEBUG]2020-06-20 05:18:43,312:utils:loss_avg: 1.05941, lr_pg0:8.035261221856173e-06, lr_pg1: 8.035261221856173e-06final_score:0.50832, mc_score:0.00917
[DEBUG]2020-06-20 05:18:43,680:utils:grad info pg0: norm std(0.007460) mean(0.021053)
[DEBUG]2020-06-20 05:18:43,683:utils:grad info pg1: norm std(0.000818) mean(0.001548)
[DEBUG]2020-06-20 05:18:43,985:utils:on_backward_begin lr: 8.317637711026711e-06
[DEBUG]2020-06-20 05:18:43,987:utils:itr: 128, num_batch: 128, last loss: 0.949209, smooth_loss: 1.015125
[DEBUG]2020-06-20 05:18:43,998:utils:loss_avg: 1.05856, lr_pg0:8.317637711026711e-06, lr_pg1: 8.317637711026711e-06final_score:0.50884, mc_score:0.00756

[DEBUG]2020-06-20 05:18:44,372:utils:grad info pg0: norm std(0.009352) mean(0.026346)
[DEBUG]2020-06-20 05:18:44,376:utils:grad info pg1: norm std(0.001102) mean(0.002180)
[DEBUG]2020-06-20 05:18:44,685:utils:on_backward_begin lr: 8.609937521846008e-06
[DEBUG]2020-06-20 05:18:44,686:utils:itr: 129, num_batch: 129, last loss: 0.945714, smooth_loss: 1.013629
[DEBUG]2020-06-20 05:18:44,696:utils:loss_avg: 1.05769, lr_pg0:8.609937521846008e-06, lr_pg1: 8.609937521846008e-06final_score:0.50996, mc_score:0.00790
[DEBUG]2020-06-20 05:18:45,064:utils:grad info pg0: norm std(0.013981) mean(0.035086)
[DEBUG]2020-06-20 05:18:45,067:utils:grad info pg1: norm std(0.001494) mean(0.002945)
[DEBUG]2020-06-20 05:18:45,371:utils:on_backward_begin lr: 8.912509381337458e-06
[DEBUG]2020-06-20 05:18:45,373:utils:itr: 130, num_batch: 130, last loss: 0.858657, smooth_loss: 1.010293
[DEBUG]2020-06-20 05:18:45,383:utils:loss_avg: 1.05617, lr_pg0:8.912509381337458e-06, lr_pg1: 8.912509381337458e-06final_score:0.51063, mc_score:0.00726
[DEBUG]2020-06-20 05:18:45,752:utils:grad info pg0: norm std(0.084953) mean(0.063274)
[DEBUG]2020-06-20 05:18:45,756:utils:grad info pg1: norm std(0.001641) mean(0.003144)
[DEBUG]2020-06-20 05:18:46,055:utils:on_backward_begin lr: 9.225714271547633e-06
[DEBUG]2020-06-20 05:18:46,056:utils:itr: 131, num_batch: 131, last loss: 0.820879, smooth_loss: 1.006222
[DEBUG]2020-06-20 05:18:46,066:utils:loss_avg: 1.05439, lr_pg0:9.225714271547633e-06, lr_pg1: 9.225714271547633e-06final_score:0.51235, mc_score:0.00982
[DEBUG]2020-06-20 05:18:46,434:utils:grad info pg0: norm std(0.010035) mean(0.016917)
[DEBUG]2020-06-20 05:18:46,436:utils:grad info pg1: norm std(0.000594) mean(0.001140)
[DEBUG]2020-06-20 05:18:46,736:utils:on_backward_begin lr: 9.549925860214362e-06
[DEBUG]2020-06-20 05:18:46,737:utils:itr: 132, num_batch: 132, last loss: 1.024146, smooth_loss: 1.006606
[DEBUG]2020-06-20 05:18:46,747:utils:loss_avg: 1.05416, lr_pg0:9.549925860214362e-06, lr_pg1: 9.549925860214362e-06final_score:0.51289, mc_score:0.00954
[DEBUG]2020-06-20 05:18:47,115:utils:grad info pg0: norm std(0.003841) mean(0.010012)
[DEBUG]2020-06-20 05:18:47,119:utils:grad info pg1: norm std(0.000565) mean(0.001019)
[DEBUG]2020-06-20 05:18:47,418:utils:on_backward_begin lr: 9.885530946569392e-06
[DEBUG]2020-06-20 05:18:47,420:utils:itr: 133, num_batch: 133, last loss: 1.064284, smooth_loss: 1.007843
[DEBUG]2020-06-20 05:18:47,430:utils:loss_avg: 1.05423, lr_pg0:9.885530946569392e-06, lr_pg1: 9.885530946569392e-06final_score:0.51090, mc_score:0.00500
[DEBUG]2020-06-20 05:18:47,798:utils:grad info pg0: norm std(0.007621) mean(0.018693)
[DEBUG]2020-06-20 05:18:47,803:utils:grad info pg1: norm std(0.000879) mean(0.001655)
[DEBUG]2020-06-20 05:18:48,115:utils:on_backward_begin lr: 1.0232929922807545e-05
[DEBUG]2020-06-20 05:18:48,116:utils:itr: 134, num_batch: 134, last loss: 0.939606, smooth_loss: 1.006383
[DEBUG]2020-06-20 05:18:48,125:utils:loss_avg: 1.05338, lr_pg0:1.0232929922807545e-05, lr_pg1: 1.0232929922807545e-05final_score:0.51045, mc_score:0.00423
[DEBUG]2020-06-20 05:18:48,494:utils:grad info pg0: norm std(0.033146) mean(0.054452)
[DEBUG]2020-06-20 05:18:48,497:utils:grad info pg1: norm std(0.002336) mean(0.004091)
[DEBUG]2020-06-20 05:18:48,797:utils:on_backward_begin lr: 1.0592537251772893e-05
[DEBUG]2020-06-20 05:18:48,799:utils:itr: 135, num_batch: 135, last loss: 0.682270, smooth_loss: 0.999456
[DEBUG]2020-06-20 05:18:48,808:utils:loss_avg: 1.05066, lr_pg0:1.0592537251772893e-05, lr_pg1: 1.0592537251772893e-05final_score:0.51181, mc_score:0.00840
[DEBUG]2020-06-20 05:18:49,181:utils:grad info pg0: norm std(0.009144) mean(0.022469)
[DEBUG]2020-06-20 05:18:49,184:utils:grad info pg1: norm std(0.001315) mean(0.002294)
[DEBUG]2020-06-20 05:18:49,494:utils:on_backward_begin lr: 1.0964781961431855e-05
[DEBUG]2020-06-20 05:18:49,496:utils:itr: 136, num_batch: 136, last loss: 0.916210, smooth_loss: 0.997680
[DEBUG]2020-06-20 05:18:49,509:utils:loss_avg: 1.04967, lr_pg0:1.0964781961431855e-05, lr_pg1: 1.0964781961431855e-05final_score:0.51218, mc_score:0.00704
[DEBUG]2020-06-20 05:18:49,879:utils:grad info pg0: norm std(0.017625) mean(0.044076)
[DEBUG]2020-06-20 05:18:49,882:utils:grad info pg1: norm std(0.001962) mean(0.003660)
[DEBUG]2020-06-20 05:18:50,180:utils:on_backward_begin lr: 1.1350108156723154e-05
[DEBUG]2020-06-20 05:18:50,181:utils:itr: 137, num_batch: 137, last loss: 0.838465, smooth_loss: 0.994287
[DEBUG]2020-06-20 05:18:50,193:utils:loss_avg: 1.04814, lr_pg0:1.1350108156723154e-05, lr_pg1: 1.1350108156723154e-05final_score:0.51241, mc_score:0.00644
[DEBUG]2020-06-20 05:18:50,567:utils:grad info pg0: norm std(0.011786) mean(0.027889)
[DEBUG]2020-06-20 05:18:50,571:utils:grad info pg1: norm std(0.001447) mean(0.002626)
[DEBUG]2020-06-20 05:18:50,886:utils:on_backward_begin lr: 1.1748975549395292e-05
[DEBUG]2020-06-20 05:18:50,887:utils:itr: 138, num_batch: 138, last loss: 0.829175, smooth_loss: 0.990773
[DEBUG]2020-06-20 05:18:50,898:utils:loss_avg: 1.04657, lr_pg0:1.1748975549395292e-05, lr_pg1: 1.1748975549395292e-05final_score:0.51284, mc_score:0.00537
[DEBUG]2020-06-20 05:18:51,266:utils:grad info pg0: norm std(0.007312) mean(0.020308)
[DEBUG]2020-06-20 05:18:51,269:utils:grad info pg1: norm std(0.001042) mean(0.001899)
[DEBUG]2020-06-20 05:18:51,569:utils:on_backward_begin lr: 1.2161860006463676e-05
[DEBUG]2020-06-20 05:18:51,571:utils:itr: 139, num_batch: 139, last loss: 0.998915, smooth_loss: 0.990946
[DEBUG]2020-06-20 05:18:51,580:utils:loss_avg: 1.04623, lr_pg0:1.2161860006463676e-05, lr_pg1: 1.2161860006463676e-05final_score:0.51243, mc_score:0.00468
[DEBUG]2020-06-20 05:18:51,946:utils:grad info pg0: norm std(0.007638) mean(0.017343)
[DEBUG]2020-06-20 05:18:51,950:utils:grad info pg1: norm std(0.000751) mean(0.001462)
[DEBUG]2020-06-20 05:18:52,250:utils:on_backward_begin lr: 1.258925411794167e-05
[DEBUG]2020-06-20 05:18:52,252:utils:itr: 140, num_batch: 140, last loss: 1.025844, smooth_loss: 0.991687
[DEBUG]2020-06-20 05:18:52,261:utils:loss_avg: 1.04608, lr_pg0:1.258925411794167e-05, lr_pg1: 1.258925411794167e-05final_score:0.51323, mc_score:0.00429
[DEBUG]2020-06-20 05:18:52,629:utils:grad info pg0: norm std(0.002421) mean(0.006147)
[DEBUG]2020-06-20 05:18:52,632:utils:grad info pg1: norm std(0.000320) mean(0.000603)
[DEBUG]2020-06-20 05:18:52,944:utils:on_backward_begin lr: 1.303166778452299e-05
[DEBUG]2020-06-20 05:18:52,945:utils:itr: 141, num_batch: 141, last loss: 1.016108, smooth_loss: 0.992204
[DEBUG]2020-06-20 05:18:52,955:utils:loss_avg: 1.04587, lr_pg0:1.303166778452299e-05, lr_pg1: 1.303166778452299e-05final_score:0.51304, mc_score:0.00287
[DEBUG]2020-06-20 05:18:53,322:utils:grad info pg0: norm std(0.007883) mean(0.019783)
[DEBUG]2020-06-20 05:18:53,325:utils:grad info pg1: norm std(0.001016) mean(0.001862)
[DEBUG]2020-06-20 05:18:53,623:utils:on_backward_begin lr: 1.3489628825916533e-05
[DEBUG]2020-06-20 05:18:53,625:utils:itr: 142, num_batch: 142, last loss: 0.652056, smooth_loss: 0.985001
[DEBUG]2020-06-20 05:18:53,634:utils:loss_avg: 1.04312, lr_pg0:1.3489628825916533e-05, lr_pg1: 1.3489628825916533e-05final_score:0.51605, mc_score:0.00786
[DEBUG]2020-06-20 05:18:54,002:utils:grad info pg0: norm std(0.011817) mean(0.034865)
[DEBUG]2020-06-20 05:18:54,005:utils:grad info pg1: norm std(0.001380) mean(0.002744)
[DEBUG]2020-06-20 05:18:54,306:utils:on_backward_begin lr: 1.3963683610559373e-05
[DEBUG]2020-06-20 05:18:54,307:utils:itr: 143, num_batch: 143, last loss: 0.978122, smooth_loss: 0.984855
[DEBUG]2020-06-20 05:18:54,316:utils:loss_avg: 1.04267, lr_pg0:1.3963683610559373e-05, lr_pg1: 1.3963683610559373e-05final_score:0.51510, mc_score:0.00599
[DEBUG]2020-06-20 05:18:54,682:utils:grad info pg0: norm std(0.005765) mean(0.015657)
[DEBUG]2020-06-20 05:18:54,686:utils:grad info pg1: norm std(0.000834) mean(0.001531)
[DEBUG]2020-06-20 05:18:54,988:utils:on_backward_begin lr: 1.4454397707459272e-05
[DEBUG]2020-06-20 05:18:54,990:utils:itr: 144, num_batch: 144, last loss: 0.923076, smooth_loss: 0.983550
[DEBUG]2020-06-20 05:18:54,999:utils:loss_avg: 1.04184, lr_pg0:1.4454397707459272e-05, lr_pg1: 1.4454397707459272e-05final_score:0.51458, mc_score:0.00729

[DEBUG]2020-06-20 05:18:55,371:utils:grad info pg0: norm std(0.014300) mean(0.033579)
[DEBUG]2020-06-20 05:18:55,375:utils:grad info pg1: norm std(0.001233) mean(0.002419)
[DEBUG]2020-06-20 05:18:55,690:utils:on_backward_begin lr: 1.496235656094433e-05
[DEBUG]2020-06-20 05:18:55,692:utils:itr: 145, num_batch: 145, last loss: 0.918116, smooth_loss: 0.982169
[DEBUG]2020-06-20 05:18:55,702:utils:loss_avg: 1.04100, lr_pg0:1.496235656094433e-05, lr_pg1: 1.496235656094433e-05final_score:0.51395, mc_score:0.00618
[DEBUG]2020-06-20 05:18:56,070:utils:grad info pg0: norm std(0.013248) mean(0.025091)
[DEBUG]2020-06-20 05:18:56,073:utils:grad info pg1: norm std(0.001160) mean(0.002244)
[DEBUG]2020-06-20 05:18:56,376:utils:on_backward_begin lr: 1.548816618912481e-05
[DEBUG]2020-06-20 05:18:56,378:utils:itr: 146, num_batch: 146, last loss: 0.887408, smooth_loss: 0.980171
[DEBUG]2020-06-20 05:18:56,387:utils:loss_avg: 1.03995, lr_pg0:1.548816618912481e-05, lr_pg1: 1.548816618912481e-05final_score:0.51363, mc_score:0.00426
[DEBUG]2020-06-20 05:18:56,760:utils:grad info pg0: norm std(0.013261) mean(0.034286)
[DEBUG]2020-06-20 05:18:56,763:utils:grad info pg1: norm std(0.001481) mean(0.002948)
[DEBUG]2020-06-20 05:18:57,060:utils:on_backward_begin lr: 1.6032453906900413e-05
[DEBUG]2020-06-20 05:18:57,061:utils:itr: 147, num_batch: 147, last loss: 0.778200, smooth_loss: 0.975918
[DEBUG]2020-06-20 05:18:57,071:utils:loss_avg: 1.03818, lr_pg0:1.6032453906900413e-05, lr_pg1: 1.6032453906900413e-05final_score:0.51583, mc_score:0.00695
[DEBUG]2020-06-20 05:18:57,441:utils:grad info pg0: norm std(0.006336) mean(0.016994)
[DEBUG]2020-06-20 05:18:57,444:utils:grad info pg1: norm std(0.000765) mean(0.001504)
[DEBUG]2020-06-20 05:18:57,742:utils:on_backward_begin lr: 1.659586907437561e-05
[DEBUG]2020-06-20 05:18:57,744:utils:itr: 148, num_batch: 148, last loss: 1.021664, smooth_loss: 0.976880
[DEBUG]2020-06-20 05:18:57,753:utils:loss_avg: 1.03807, lr_pg0:1.659586907437561e-05, lr_pg1: 1.659586907437561e-05final_score:0.51554, mc_score:0.00485
[DEBUG]2020-06-20 05:18:58,120:utils:grad info pg0: norm std(0.021617) mean(0.046232)
[DEBUG]2020-06-20 05:18:58,123:utils:grad info pg1: norm std(0.002066) mean(0.003851)
[DEBUG]2020-06-20 05:18:58,423:utils:on_backward_begin lr: 1.717908387157588e-05
[DEBUG]2020-06-20 05:18:58,425:utils:itr: 149, num_batch: 149, last loss: 0.857825, smooth_loss: 0.974378
[DEBUG]2020-06-20 05:18:58,435:utils:loss_avg: 1.03687, lr_pg0:1.717908387157588e-05, lr_pg1: 1.717908387157588e-05final_score:0.51682, mc_score:0.00781
[DEBUG]2020-06-20 05:18:58,803:utils:grad info pg0: norm std(0.010778) mean(0.030300)
[DEBUG]2020-06-20 05:18:58,806:utils:grad info pg1: norm std(0.001095) mean(0.002210)
[DEBUG]2020-06-20 05:18:59,111:utils:on_backward_begin lr: 1.778279410038923e-05
[DEBUG]2020-06-20 05:18:59,112:utils:itr: 150, num_batch: 150, last loss: 1.186603, smooth_loss: 0.978834
[DEBUG]2020-06-20 05:18:59,122:utils:loss_avg: 1.03786, lr_pg0:1.778279410038923e-05, lr_pg1: 1.778279410038923e-05final_score:0.51318, mc_score:0.00101
[DEBUG]2020-06-20 05:18:59,491:utils:grad info pg0: norm std(0.017106) mean(0.032365)
[DEBUG]2020-06-20 05:18:59,495:utils:grad info pg1: norm std(0.001710) mean(0.003348)
[DEBUG]2020-06-20 05:18:59,792:utils:on_backward_begin lr: 1.840772001468956e-05
[DEBUG]2020-06-20 05:18:59,794:utils:itr: 151, num_batch: 151, last loss: 1.011488, smooth_loss: 0.979519
[DEBUG]2020-06-20 05:18:59,803:utils:loss_avg: 1.03769, lr_pg0:1.840772001468956e-05, lr_pg1: 1.840772001468956e-05final_score:0.51411, mc_score:0.00126
[DEBUG]2020-06-20 05:19:00,169:utils:grad info pg0: norm std(0.025612) mean(0.043037)
[DEBUG]2020-06-20 05:19:00,173:utils:grad info pg1: norm std(0.001924) mean(0.003718)
[DEBUG]2020-06-20 05:19:00,474:utils:on_backward_begin lr: 1.9054607179632474e-05
[DEBUG]2020-06-20 05:19:00,475:utils:itr: 152, num_batch: 152, last loss: 0.729880, smooth_loss: 0.974288
[DEBUG]2020-06-20 05:19:00,484:utils:loss_avg: 1.03568, lr_pg0:1.9054607179632474e-05, lr_pg1: 1.9054607179632474e-05final_score:0.51555, mc_score:0.00205
[DEBUG]2020-06-20 05:19:00,852:utils:grad info pg0: norm std(0.037641) mean(0.084972)
[DEBUG]2020-06-20 05:19:00,857:utils:grad info pg1: norm std(0.003297) mean(0.006543)
[DEBUG]2020-06-20 05:19:01,161:utils:on_backward_begin lr: 1.972422736114854e-05
[DEBUG]2020-06-20 05:19:01,162:utils:itr: 153, num_batch: 153, last loss: 0.990047, smooth_loss: 0.974618
[DEBUG]2020-06-20 05:19:01,172:utils:loss_avg: 1.03538, lr_pg0:1.972422736114854e-05, lr_pg1: 1.972422736114854e-05final_score:0.51532, mc_score:0.00238
[DEBUG]2020-06-20 05:19:01,539:utils:grad info pg0: norm std(0.021671) mean(0.042346)
[DEBUG]2020-06-20 05:19:01,543:utils:grad info pg1: norm std(0.002057) mean(0.003767)
[DEBUG]2020-06-20 05:19:01,846:utils:on_backward_begin lr: 2.0417379446695298e-05
[DEBUG]2020-06-20 05:19:01,847:utils:itr: 154, num_batch: 154, last loss: 0.838123, smooth_loss: 0.971763
[DEBUG]2020-06-20 05:19:01,857:utils:loss_avg: 1.03411, lr_pg0:2.0417379446695298e-05, lr_pg1: 2.0417379446695298e-05final_score:0.51708, mc_score:0.00367
[DEBUG]2020-06-20 05:19:02,225:utils:grad info pg0: norm std(0.007716) mean(0.020723)
[DEBUG]2020-06-20 05:19:02,228:utils:grad info pg1: norm std(0.000984) mean(0.001841)
[DEBUG]2020-06-20 05:19:02,530:utils:on_backward_begin lr: 2.1134890398366472e-05
[DEBUG]2020-06-20 05:19:02,531:utils:itr: 155, num_batch: 155, last loss: 0.972963, smooth_loss: 0.971789
[DEBUG]2020-06-20 05:19:02,540:utils:loss_avg: 1.03371, lr_pg0:2.1134890398366472e-05, lr_pg1: 2.1134890398366472e-05final_score:0.51767, mc_score:0.00444
[DEBUG]2020-06-20 05:19:02,910:utils:grad info pg0: norm std(0.011411) mean(0.030743)
[DEBUG]2020-06-20 05:19:02,915:utils:grad info pg1: norm std(0.001636) mean(0.003102)
[DEBUG]2020-06-20 05:19:03,226:utils:on_backward_begin lr: 2.1877616239495533e-05
[DEBUG]2020-06-20 05:19:03,227:utils:itr: 156, num_batch: 156, last loss: 0.632084, smooth_loss: 0.964697
[DEBUG]2020-06-20 05:19:03,238:utils:loss_avg: 1.03116, lr_pg0:2.1877616239495533e-05, lr_pg1: 2.1877616239495533e-05final_score:0.52023, mc_score:0.00887
[DEBUG]2020-06-20 05:19:03,612:utils:grad info pg0: norm std(0.002483) mean(0.007980)
[DEBUG]2020-06-20 05:19:03,616:utils:grad info pg1: norm std(0.000436) mean(0.000786)
[DEBUG]2020-06-20 05:19:03,914:utils:on_backward_begin lr: 2.2646443075930603e-05
[DEBUG]2020-06-20 05:19:03,917:utils:itr: 157, num_batch: 157, last loss: 0.686036, smooth_loss: 0.958885
[DEBUG]2020-06-20 05:19:03,927:utils:loss_avg: 1.02897, lr_pg0:2.2646443075930603e-05, lr_pg1: 2.2646443075930603e-05final_score:0.52315, mc_score:0.01322
[DEBUG]2020-06-20 05:19:04,296:utils:grad info pg0: norm std(0.009081) mean(0.022964)
[DEBUG]2020-06-20 05:19:04,299:utils:grad info pg1: norm std(0.001074) mean(0.002119)
[DEBUG]2020-06-20 05:19:04,600:utils:on_backward_begin lr: 2.344228815319923e-05
[DEBUG]2020-06-20 05:19:04,601:utils:itr: 158, num_batch: 158, last loss: 0.721573, smooth_loss: 0.953940
[DEBUG]2020-06-20 05:19:04,611:utils:loss_avg: 1.02704, lr_pg0:2.344228815319923e-05, lr_pg1: 2.344228815319923e-05final_score:0.52497, mc_score:0.01652
[DEBUG]2020-06-20 05:19:04,981:utils:grad info pg0: norm std(0.007496) mean(0.022797)
[DEBUG]2020-06-20 05:19:04,984:utils:grad info pg1: norm std(0.000983) mean(0.001901)
[DEBUG]2020-06-20 05:19:05,285:utils:on_backward_begin lr: 2.4266100950824165e-05
[DEBUG]2020-06-20 05:19:05,286:utils:itr: 159, num_batch: 159, last loss: 1.091335, smooth_loss: 0.956801
[DEBUG]2020-06-20 05:19:05,297:utils:loss_avg: 1.02744, lr_pg0:2.4266100950824165e-05, lr_pg1: 2.4266100950824165e-05final_score:0.52319, mc_score:0.01375
[DEBUG]2020-06-20 05:19:05,671:utils:grad info pg0: norm std(0.012579) mean(0.033842)
[DEBUG]2020-06-20 05:19:05,674:utils:grad info pg1: norm std(0.001691) mean(0.003240)
[DEBUG]2020-06-20 05:19:05,974:utils:on_backward_begin lr: 2.5118864315095812e-05
[DEBUG]2020-06-20 05:19:05,975:utils:itr: 160, num_batch: 160, last loss: 1.010024, smooth_loss: 0.957908
[DEBUG]2020-06-20 05:19:05,985:utils:loss_avg: 1.02733, lr_pg0:2.5118864315095812e-05, lr_pg1: 2.5118864315095812e-05final_score:0.52239, mc_score:0.01150

[DEBUG]2020-06-20 05:19:06,354:utils:grad info pg0: norm std(0.006783) mean(0.017282)
[DEBUG]2020-06-20 05:19:06,357:utils:grad info pg1: norm std(0.000970) mean(0.001757)
[DEBUG]2020-06-20 05:19:06,656:utils:on_backward_begin lr: 2.600159563165273e-05
[DEBUG]2020-06-20 05:19:06,657:utils:itr: 161, num_batch: 161, last loss: 0.638181, smooth_loss: 0.951261
[DEBUG]2020-06-20 05:19:06,668:utils:loss_avg: 1.02493, lr_pg0:2.600159563165273e-05, lr_pg1: 2.600159563165273e-05final_score:0.52473, mc_score:0.01412
[DEBUG]2020-06-20 05:19:07,040:utils:grad info pg0: norm std(0.007167) mean(0.016302)
[DEBUG]2020-06-20 05:19:07,043:utils:grad info pg1: norm std(0.000811) mean(0.001534)
[DEBUG]2020-06-20 05:19:07,341:utils:on_backward_begin lr: 2.6915348039269167e-05
[DEBUG]2020-06-20 05:19:07,342:utils:itr: 162, num_batch: 162, last loss: 0.920838, smooth_loss: 0.950629
[DEBUG]2020-06-20 05:19:07,353:utils:loss_avg: 1.02429, lr_pg0:2.6915348039269167e-05, lr_pg1: 2.6915348039269167e-05final_score:0.52699, mc_score:0.01623
[DEBUG]2020-06-20 05:19:07,721:utils:grad info pg0: norm std(0.010740) mean(0.028570)
[DEBUG]2020-06-20 05:19:07,724:utils:grad info pg1: norm std(0.001143) mean(0.002337)
[DEBUG]2020-06-20 05:19:08,024:utils:on_backward_begin lr: 2.7861211686297695e-05
[DEBUG]2020-06-20 05:19:08,026:utils:itr: 163, num_batch: 163, last loss: 0.604349, smooth_loss: 0.943442
[DEBUG]2020-06-20 05:19:08,035:utils:loss_avg: 1.02173, lr_pg0:2.7861211686297695e-05, lr_pg1: 2.7861211686297695e-05final_score:0.52963, mc_score:0.02204
[DEBUG]2020-06-20 05:19:08,403:utils:grad info pg0: norm std(0.019351) mean(0.035507)
[DEBUG]2020-06-20 05:19:08,406:utils:grad info pg1: norm std(0.001912) mean(0.003487)
[DEBUG]2020-06-20 05:19:08,703:utils:on_backward_begin lr: 2.884031503126605e-05
[DEBUG]2020-06-20 05:19:08,705:utils:itr: 164, num_batch: 164, last loss: 0.698248, smooth_loss: 0.938357
[DEBUG]2020-06-20 05:19:08,714:utils:loss_avg: 1.01977, lr_pg0:2.884031503126605e-05, lr_pg1: 2.884031503126605e-05final_score:0.53296, mc_score:0.02688
[DEBUG]2020-06-20 05:19:09,081:utils:grad info pg0: norm std(0.003599) mean(0.009713)
[DEBUG]2020-06-20 05:19:09,085:utils:grad info pg1: norm std(0.000739) mean(0.001148)
[DEBUG]2020-06-20 05:19:09,386:utils:on_backward_begin lr: 2.985382618917959e-05
[DEBUG]2020-06-20 05:19:09,388:utils:itr: 165, num_batch: 165, last loss: 0.793642, smooth_loss: 0.935358
[DEBUG]2020-06-20 05:19:09,398:utils:loss_avg: 1.01841, lr_pg0:2.985382618917959e-05, lr_pg1: 2.985382618917959e-05final_score:0.53432, mc_score:0.02679
[DEBUG]2020-06-20 05:19:09,765:utils:grad info pg0: norm std(0.016688) mean(0.035164)
[DEBUG]2020-06-20 05:19:09,769:utils:grad info pg1: norm std(0.001794) mean(0.003249)
[DEBUG]2020-06-20 05:19:10,261:utils:on_backward_begin lr: 3.0902954325135894e-05
[DEBUG]2020-06-20 05:19:10,262:utils:itr: 166, num_batch: 166, last loss: 0.798350, smooth_loss: 0.932521
[DEBUG]2020-06-20 05:19:10,272:utils:loss_avg: 1.01709, lr_pg0:3.0902954325135894e-05, lr_pg1: 3.0902954325135894e-05final_score:0.53504, mc_score:0.02833
[DEBUG]2020-06-20 05:19:10,645:utils:grad info pg0: norm std(0.021861) mean(0.041719)
[DEBUG]2020-06-20 05:19:10,648:utils:grad info pg1: norm std(0.001868) mean(0.003504)
[DEBUG]2020-06-20 05:19:10,950:utils:on_backward_begin lr: 3.1988951096913973e-05
[DEBUG]2020-06-20 05:19:10,952:utils:itr: 167, num_batch: 167, last loss: 0.891872, smooth_loss: 0.931679
[DEBUG]2020-06-20 05:19:10,962:utils:loss_avg: 1.01635, lr_pg0:3.1988951096913973e-05, lr_pg1: 3.1988951096913973e-05final_score:0.53725, mc_score:0.03130
[DEBUG]2020-06-20 05:19:11,335:utils:grad info pg0: norm std(0.007784) mean(0.019917)
[DEBUG]2020-06-20 05:19:11,339:utils:grad info pg1: norm std(0.001285) mean(0.002132)
[DEBUG]2020-06-20 05:19:11,652:utils:on_backward_begin lr: 3.3113112148259103e-05
[DEBUG]2020-06-20 05:19:11,654:utils:itr: 168, num_batch: 168, last loss: 0.915453, smooth_loss: 0.931344
[DEBUG]2020-06-20 05:19:11,664:utils:loss_avg: 1.01575, lr_pg0:3.3113112148259103e-05, lr_pg1: 3.3113112148259103e-05final_score:0.53949, mc_score:0.03396
[DEBUG]2020-06-20 05:19:12,035:utils:grad info pg0: norm std(0.004892) mean(0.014320)
[DEBUG]2020-06-20 05:19:12,039:utils:grad info pg1: norm std(0.000919) mean(0.001620)
[DEBUG]2020-06-20 05:19:12,355:utils:on_backward_begin lr: 3.427677865464503e-05
[DEBUG]2020-06-20 05:19:12,357:utils:itr: 169, num_batch: 169, last loss: 0.706289, smooth_loss: 0.926693
[DEBUG]2020-06-20 05:19:12,367:utils:loss_avg: 1.01393, lr_pg0:3.427677865464503e-05, lr_pg1: 3.427677865464503e-05final_score:0.54282, mc_score:0.03815
[DEBUG]2020-06-20 05:19:12,740:utils:grad info pg0: norm std(0.007743) mean(0.020586)
[DEBUG]2020-06-20 05:19:12,744:utils:grad info pg1: norm std(0.001167) mean(0.001992)
[DEBUG]2020-06-20 05:19:13,045:utils:on_backward_begin lr: 3.548133892335754e-05
[DEBUG]2020-06-20 05:19:13,046:utils:itr: 170, num_batch: 170, last loss: 0.678658, smooth_loss: 0.921570
[DEBUG]2020-06-20 05:19:13,056:utils:loss_avg: 1.01197, lr_pg0:3.548133892335754e-05, lr_pg1: 3.548133892335754e-05final_score:0.54523, mc_score:0.04075
[DEBUG]2020-06-20 05:19:13,424:utils:grad info pg0: norm std(0.004084) mean(0.009973)
[DEBUG]2020-06-20 05:19:13,428:utils:grad info pg1: norm std(0.000762) mean(0.001244)
[DEBUG]2020-06-20 05:19:13,726:utils:on_backward_begin lr: 3.6728230049808465e-05
[DEBUG]2020-06-20 05:19:13,727:utils:itr: 171, num_batch: 171, last loss: 0.510026, smooth_loss: 0.913076
[DEBUG]2020-06-20 05:19:13,740:utils:loss_avg: 1.00905, lr_pg0:3.6728230049808465e-05, lr_pg1: 3.6728230049808465e-05final_score:0.54892, mc_score:0.04615
[DEBUG]2020-06-20 05:19:14,107:utils:grad info pg0: norm std(0.028168) mean(0.053495)
[DEBUG]2020-06-20 05:19:14,110:utils:grad info pg1: norm std(0.002914) mean(0.004550)
[DEBUG]2020-06-20 05:19:14,410:utils:on_backward_begin lr: 3.801893963205612e-05
[DEBUG]2020-06-20 05:19:14,412:utils:itr: 172, num_batch: 172, last loss: 0.998470, smooth_loss: 0.914838
[DEBUG]2020-06-20 05:19:14,422:utils:loss_avg: 1.00899, lr_pg0:3.801893963205612e-05, lr_pg1: 3.801893963205612e-05final_score:0.54983, mc_score:0.04802
[DEBUG]2020-06-20 05:19:14,797:utils:grad info pg0: norm std(0.006711) mean(0.016397)
[DEBUG]2020-06-20 05:19:14,801:utils:grad info pg1: norm std(0.001319) mean(0.002161)
[DEBUG]2020-06-20 05:19:15,102:utils:on_backward_begin lr: 3.935500754557775e-05
[DEBUG]2020-06-20 05:19:15,104:utils:itr: 173, num_batch: 173, last loss: 0.891770, smooth_loss: 0.914362
[DEBUG]2020-06-20 05:19:15,114:utils:loss_avg: 1.00831, lr_pg0:3.935500754557775e-05, lr_pg1: 3.935500754557775e-05final_score:0.55150, mc_score:0.04897
[DEBUG]2020-06-20 05:19:15,483:utils:grad info pg0: norm std(0.007104) mean(0.015175)
[DEBUG]2020-06-20 05:19:15,487:utils:grad info pg1: norm std(0.000897) mean(0.001595)
[DEBUG]2020-06-20 05:19:15,783:utils:on_backward_begin lr: 4.0738027780411274e-05
[DEBUG]2020-06-20 05:19:15,785:utils:itr: 174, num_batch: 174, last loss: 0.533167, smooth_loss: 0.906510
[DEBUG]2020-06-20 05:19:15,795:utils:loss_avg: 1.00560, lr_pg0:4.0738027780411274e-05, lr_pg1: 4.0738027780411274e-05final_score:0.55638, mc_score:0.05474
[DEBUG]2020-06-20 05:19:16,162:utils:grad info pg0: norm std(0.005745) mean(0.012969)
[DEBUG]2020-06-20 05:19:16,166:utils:grad info pg1: norm std(0.000743) mean(0.001359)
[DEBUG]2020-06-20 05:19:16,465:utils:on_backward_begin lr: 4.216965034285822e-05
[DEBUG]2020-06-20 05:19:16,467:utils:itr: 175, num_batch: 175, last loss: 0.882636, smooth_loss: 0.906018
[DEBUG]2020-06-20 05:19:16,476:utils:loss_avg: 1.00490, lr_pg0:4.216965034285822e-05, lr_pg1: 4.216965034285822e-05final_score:0.55793, mc_score:0.05690
[DEBUG]2020-06-20 05:19:16,846:utils:grad info pg0: norm std(0.011981) mean(0.028810)
[DEBUG]2020-06-20 05:19:16,850:utils:grad info pg1: norm std(0.001303) mean(0.002450)
[DEBUG]2020-06-20 05:19:17,150:utils:on_backward_begin lr: 4.36515832240166e-05
[DEBUG]2020-06-20 05:19:17,151:utils:itr: 176, num_batch: 176, last loss: 0.482069, smooth_loss: 0.897295
[DEBUG]2020-06-20 05:19:17,161:utils:loss_avg: 1.00195, lr_pg0:4.36515832240166e-05, lr_pg1: 4.36515832240166e-05final_score:0.56265, mc_score:0.06301

[DEBUG]2020-06-20 05:19:17,529:utils:grad info pg0: norm std(0.004777) mean(0.012327)
[DEBUG]2020-06-20 05:19:17,533:utils:grad info pg1: norm std(0.000612) mean(0.001135)
[DEBUG]2020-06-20 05:19:17,833:utils:on_backward_begin lr: 4.518559443749224e-05
[DEBUG]2020-06-20 05:19:17,835:utils:itr: 177, num_batch: 177, last loss: 0.675326, smooth_loss: 0.892730
[DEBUG]2020-06-20 05:19:17,846:utils:loss_avg: 1.00011, lr_pg0:4.518559443749224e-05, lr_pg1: 4.518559443749224e-05final_score:0.56530, mc_score:0.06705
[DEBUG]2020-06-20 05:19:18,218:utils:grad info pg0: norm std(0.007148) mean(0.020115)
[DEBUG]2020-06-20 05:19:18,222:utils:grad info pg1: norm std(0.001141) mean(0.002041)
[DEBUG]2020-06-20 05:19:18,524:utils:on_backward_begin lr: 4.677351412871983e-05
[DEBUG]2020-06-20 05:19:18,526:utils:itr: 178, num_batch: 178, last loss: 1.044812, smooth_loss: 0.895856
[DEBUG]2020-06-20 05:19:18,536:utils:loss_avg: 1.00036, lr_pg0:4.677351412871983e-05, lr_pg1: 4.677351412871983e-05final_score:0.56451, mc_score:0.06567
[DEBUG]2020-06-20 05:19:18,909:utils:grad info pg0: norm std(0.005447) mean(0.015256)
[DEBUG]2020-06-20 05:19:18,912:utils:grad info pg1: norm std(0.000763) mean(0.001438)
[DEBUG]2020-06-20 05:19:19,209:utils:on_backward_begin lr: 4.841723675840994e-05
[DEBUG]2020-06-20 05:19:19,211:utils:itr: 179, num_batch: 179, last loss: 0.692116, smooth_loss: 0.891671
[DEBUG]2020-06-20 05:19:19,225:utils:loss_avg: 0.99865, lr_pg0:4.841723675840994e-05, lr_pg1: 4.841723675840994e-05final_score:0.56644, mc_score:0.06673
[DEBUG]2020-06-20 05:19:19,594:utils:grad info pg0: norm std(0.017083) mean(0.040913)
[DEBUG]2020-06-20 05:19:19,597:utils:grad info pg1: norm std(0.001920) mean(0.003650)
[DEBUG]2020-06-20 05:19:19,909:utils:on_backward_begin lr: 5.011872336272724e-05
[DEBUG]2020-06-20 05:19:19,911:utils:itr: 180, num_batch: 180, last loss: 0.560320, smooth_loss: 0.884868
[DEBUG]2020-06-20 05:19:19,921:utils:loss_avg: 0.99623, lr_pg0:5.011872336272724e-05, lr_pg1: 5.011872336272724e-05final_score:0.56994, mc_score:0.06994
[DEBUG]2020-06-20 05:19:20,295:utils:grad info pg0: norm std(0.004962) mean(0.016119)
[DEBUG]2020-06-20 05:19:20,299:utils:grad info pg1: norm std(0.000797) mean(0.001464)
[DEBUG]2020-06-20 05:19:20,607:utils:on_backward_begin lr: 5.188000389289612e-05
[DEBUG]2020-06-20 05:19:20,609:utils:itr: 181, num_batch: 181, last loss: 0.801274, smooth_loss: 0.883153
[DEBUG]2020-06-20 05:19:20,619:utils:loss_avg: 0.99516, lr_pg0:5.188000389289612e-05, lr_pg1: 5.188000389289612e-05final_score:0.57138, mc_score:0.06975
[DEBUG]2020-06-20 05:19:20,986:utils:grad info pg0: norm std(0.004084) mean(0.010855)
[DEBUG]2020-06-20 05:19:20,989:utils:grad info pg1: norm std(0.000633) mean(0.001098)
[DEBUG]2020-06-20 05:19:21,293:utils:on_backward_begin lr: 5.3703179637025284e-05
[DEBUG]2020-06-20 05:19:21,295:utils:itr: 182, num_batch: 182, last loss: 1.034991, smooth_loss: 0.886267
[DEBUG]2020-06-20 05:19:21,305:utils:loss_avg: 0.99537, lr_pg0:5.3703179637025284e-05, lr_pg1: 5.3703179637025284e-05final_score:0.57086, mc_score:0.06957
[DEBUG]2020-06-20 05:19:21,674:utils:grad info pg0: norm std(0.004400) mean(0.010522)
[DEBUG]2020-06-20 05:19:21,677:utils:grad info pg1: norm std(0.000672) mean(0.001147)
[DEBUG]2020-06-20 05:19:21,976:utils:on_backward_begin lr: 5.5590425727040367e-05
[DEBUG]2020-06-20 05:19:21,977:utils:itr: 183, num_batch: 183, last loss: 0.813029, smooth_loss: 0.884766
[DEBUG]2020-06-20 05:19:21,987:utils:loss_avg: 0.99438, lr_pg0:5.5590425727040367e-05, lr_pg1: 5.5590425727040367e-05final_score:0.57154, mc_score:0.07192
[DEBUG]2020-06-20 05:19:22,357:utils:grad info pg0: norm std(0.012186) mean(0.028893)
[DEBUG]2020-06-20 05:19:22,362:utils:grad info pg1: norm std(0.001739) mean(0.003143)
[DEBUG]2020-06-20 05:19:22,665:utils:on_backward_begin lr: 5.754399373371572e-05
[DEBUG]2020-06-20 05:19:22,667:utils:itr: 184, num_batch: 184, last loss: 0.887742, smooth_loss: 0.884827
[DEBUG]2020-06-20 05:19:22,676:utils:loss_avg: 0.99381, lr_pg0:5.754399373371572e-05, lr_pg1: 5.754399373371572e-05final_score:0.57093, mc_score:0.06836
[DEBUG]2020-06-20 05:19:23,044:utils:grad info pg0: norm std(0.009569) mean(0.022145)
[DEBUG]2020-06-20 05:19:23,048:utils:grad info pg1: norm std(0.001594) mean(0.002517)
[DEBUG]2020-06-20 05:19:23,363:utils:on_backward_begin lr: 5.956621435290106e-05
[DEBUG]2020-06-20 05:19:23,364:utils:itr: 185, num_batch: 185, last loss: 0.722530, smooth_loss: 0.881503
[DEBUG]2020-06-20 05:19:23,374:utils:loss_avg: 0.99235, lr_pg0:5.956621435290106e-05, lr_pg1: 5.956621435290106e-05final_score:0.57228, mc_score:0.07092
[DEBUG]2020-06-20 05:19:23,742:utils:grad info pg0: norm std(0.006383) mean(0.014196)
[DEBUG]2020-06-20 05:19:23,745:utils:grad info pg1: norm std(0.000790) mean(0.001472)
[DEBUG]2020-06-20 05:19:24,047:utils:on_backward_begin lr: 6.165950018614824e-05
[DEBUG]2020-06-20 05:19:24,049:utils:itr: 186, num_batch: 186, last loss: 0.836158, smooth_loss: 0.880575
[DEBUG]2020-06-20 05:19:24,060:utils:loss_avg: 0.99151, lr_pg0:6.165950018614824e-05, lr_pg1: 6.165950018614824e-05final_score:0.57214, mc_score:0.07145
[DEBUG]2020-06-20 05:19:24,428:utils:grad info pg0: norm std(0.033421) mean(0.070794)
[DEBUG]2020-06-20 05:19:24,431:utils:grad info pg1: norm std(0.003766) mean(0.006161)
[DEBUG]2020-06-20 05:19:24,731:utils:on_backward_begin lr: 6.38263486190549e-05
[DEBUG]2020-06-20 05:19:24,733:utils:itr: 187, num_batch: 187, last loss: 1.089763, smooth_loss: 0.884855
[DEBUG]2020-06-20 05:19:24,743:utils:loss_avg: 0.99204, lr_pg0:6.38263486190549e-05, lr_pg1: 6.38263486190549e-05final_score:0.57132, mc_score:0.07079
[DEBUG]2020-06-20 05:19:25,111:utils:grad info pg0: norm std(0.011630) mean(0.028978)
[DEBUG]2020-06-20 05:19:25,113:utils:grad info pg1: norm std(0.001291) mean(0.002470)
[DEBUG]2020-06-20 05:19:25,415:utils:on_backward_begin lr: 6.606934480075958e-05
[DEBUG]2020-06-20 05:19:25,417:utils:itr: 188, num_batch: 188, last loss: 0.721067, smooth_loss: 0.881506
[DEBUG]2020-06-20 05:19:25,427:utils:loss_avg: 0.99060, lr_pg0:6.606934480075958e-05, lr_pg1: 6.606934480075958e-05final_score:0.57227, mc_score:0.07317
[DEBUG]2020-06-20 05:19:25,796:utils:grad info pg0: norm std(0.023369) mean(0.036377)
[DEBUG]2020-06-20 05:19:25,799:utils:grad info pg1: norm std(0.001864) mean(0.003442)
[DEBUG]2020-06-20 05:19:26,098:utils:on_backward_begin lr: 6.839116472814291e-05
[DEBUG]2020-06-20 05:19:26,099:utils:itr: 189, num_batch: 189, last loss: 0.987563, smooth_loss: 0.883673
[DEBUG]2020-06-20 05:19:26,108:utils:loss_avg: 0.99059, lr_pg0:6.839116472814291e-05, lr_pg1: 6.839116472814291e-05final_score:0.57129, mc_score:0.07274
[DEBUG]2020-06-20 05:19:26,476:utils:grad info pg0: norm std(0.003224) mean(0.008501)
[DEBUG]2020-06-20 05:19:26,479:utils:grad info pg1: norm std(0.000394) mean(0.000750)
[DEBUG]2020-06-20 05:19:26,778:utils:on_backward_begin lr: 7.079457843841379e-05
[DEBUG]2020-06-20 05:19:26,780:utils:itr: 190, num_batch: 190, last loss: 1.075185, smooth_loss: 0.887586
[DEBUG]2020-06-20 05:19:26,790:utils:loss_avg: 0.99103, lr_pg0:7.079457843841379e-05, lr_pg1: 7.079457843841379e-05final_score:0.57000, mc_score:0.07061
[DEBUG]2020-06-20 05:19:27,158:utils:grad info pg0: norm std(0.020153) mean(0.016551)
[DEBUG]2020-06-20 05:19:27,161:utils:grad info pg1: norm std(0.000615) mean(0.001019)
[DEBUG]2020-06-20 05:19:27,461:utils:on_backward_begin lr: 7.328245331389039e-05
[DEBUG]2020-06-20 05:19:27,463:utils:itr: 191, num_batch: 191, last loss: 1.020142, smooth_loss: 0.890293
[DEBUG]2020-06-20 05:19:27,473:utils:loss_avg: 0.99118, lr_pg0:7.328245331389039e-05, lr_pg1: 7.328245331389039e-05final_score:0.56876, mc_score:0.06852
[DEBUG]2020-06-20 05:19:27,848:utils:grad info pg0: norm std(0.016640) mean(0.018794)
[DEBUG]2020-06-20 05:19:27,852:utils:grad info pg1: norm std(0.000671) mean(0.001264)
[DEBUG]2020-06-20 05:19:28,157:utils:on_backward_begin lr: 7.585775750291836e-05
[DEBUG]2020-06-20 05:19:28,159:utils:itr: 192, num_batch: 192, last loss: 0.799015, smooth_loss: 0.888430
[DEBUG]2020-06-20 05:19:28,168:utils:loss_avg: 0.99018, lr_pg0:7.585775750291836e-05, lr_pg1: 7.585775750291836e-05final_score:0.56933, mc_score:0.07194

[DEBUG]2020-06-20 05:19:28,535:utils:grad info pg0: norm std(0.008862) mean(0.018722)
[DEBUG]2020-06-20 05:19:28,538:utils:grad info pg1: norm std(0.000931) mean(0.001720)
[DEBUG]2020-06-20 05:19:28,850:utils:on_backward_begin lr: 7.852356346100716e-05
[DEBUG]2020-06-20 05:19:28,852:utils:itr: 193, num_batch: 193, last loss: 0.891834, smooth_loss: 0.888499
[DEBUG]2020-06-20 05:19:28,861:utils:loss_avg: 0.98968, lr_pg0:7.852356346100716e-05, lr_pg1: 7.852356346100716e-05final_score:0.56884, mc_score:0.07037
[DEBUG]2020-06-20 05:19:29,236:utils:grad info pg0: norm std(0.009956) mean(0.023631)
[DEBUG]2020-06-20 05:19:29,239:utils:grad info pg1: norm std(0.000967) mean(0.001843)
[DEBUG]2020-06-20 05:19:29,550:utils:on_backward_begin lr: 8.128305161640992e-05
[DEBUG]2020-06-20 05:19:29,552:utils:itr: 194, num_batch: 194, last loss: 0.871906, smooth_loss: 0.888161
[DEBUG]2020-06-20 05:19:29,564:utils:loss_avg: 0.98907, lr_pg0:8.128305161640992e-05, lr_pg1: 8.128305161640992e-05final_score:0.56974, mc_score:0.07346
[DEBUG]2020-06-20 05:19:29,932:utils:grad info pg0: norm std(0.007842) mean(0.019457)
[DEBUG]2020-06-20 05:19:29,936:utils:grad info pg1: norm std(0.000957) mean(0.001807)
[DEBUG]2020-06-20 05:19:30,238:utils:on_backward_begin lr: 8.41395141645195e-05
[DEBUG]2020-06-20 05:19:30,239:utils:itr: 195, num_batch: 195, last loss: 1.084761, smooth_loss: 0.892169
[DEBUG]2020-06-20 05:19:30,249:utils:loss_avg: 0.98956, lr_pg0:8.41395141645195e-05, lr_pg1: 8.41395141645195e-05final_score:0.57017, mc_score:0.07435
[DEBUG]2020-06-20 05:19:30,623:utils:grad info pg0: norm std(0.007011) mean(0.014896)
[DEBUG]2020-06-20 05:19:30,626:utils:grad info pg1: norm std(0.000779) mean(0.001418)
[DEBUG]2020-06-20 05:19:30,931:utils:on_backward_begin lr: 8.709635899560807e-05
[DEBUG]2020-06-20 05:19:30,932:utils:itr: 196, num_batch: 196, last loss: 1.119738, smooth_loss: 0.896808
[DEBUG]2020-06-20 05:19:30,944:utils:loss_avg: 0.99022, lr_pg0:8.709635899560807e-05, lr_pg1: 8.709635899560807e-05final_score:0.56948, mc_score:0.07371
[DEBUG]2020-06-20 05:19:31,313:utils:grad info pg0: norm std(0.003051) mean(0.008005)
[DEBUG]2020-06-20 05:19:31,316:utils:grad info pg1: norm std(0.000395) mean(0.000720)
[DEBUG]2020-06-20 05:19:31,616:utils:on_backward_begin lr: 9.015711376059568e-05
[DEBUG]2020-06-20 05:19:31,618:utils:itr: 197, num_batch: 197, last loss: 0.761763, smooth_loss: 0.894056
[DEBUG]2020-06-20 05:19:31,628:utils:loss_avg: 0.98907, lr_pg0:9.015711376059568e-05, lr_pg1: 9.015711376059568e-05final_score:0.57062, mc_score:0.07603
[DEBUG]2020-06-20 05:19:32,002:utils:grad info pg0: norm std(0.006528) mean(0.016350)
[DEBUG]2020-06-20 05:19:32,005:utils:grad info pg1: norm std(0.000846) mean(0.001576)
[DEBUG]2020-06-20 05:19:32,322:utils:on_backward_begin lr: 9.33254300796991e-05
[DEBUG]2020-06-20 05:19:32,323:utils:itr: 198, num_batch: 198, last loss: 0.822832, smooth_loss: 0.892606
[DEBUG]2020-06-20 05:19:32,335:utils:loss_avg: 0.98823, lr_pg0:9.33254300796991e-05, lr_pg1: 9.33254300796991e-05final_score:0.57037, mc_score:0.07556
[DEBUG]2020-06-20 05:19:32,706:utils:grad info pg0: norm std(0.005529) mean(0.013477)
[DEBUG]2020-06-20 05:19:32,709:utils:grad info pg1: norm std(0.000596) mean(0.001138)
[DEBUG]2020-06-20 05:19:33,017:utils:on_backward_begin lr: 9.660508789898133e-05
[DEBUG]2020-06-20 05:19:33,019:utils:itr: 199, num_batch: 199, last loss: 0.748329, smooth_loss: 0.889669
[DEBUG]2020-06-20 05:19:33,030:utils:loss_avg: 0.98703, lr_pg0:9.660508789898133e-05, lr_pg1: 9.660508789898133e-05final_score:0.57198, mc_score:0.07744
[DEBUG]2020-06-20 05:19:33,397:utils:grad info pg0: norm std(0.005991) mean(0.015973)
[DEBUG]2020-06-20 05:19:33,401:utils:grad info pg1: norm std(0.000766) mean(0.001450)
```
```
[DEBUG]2020-06-20 05:28:40,782:utils:on_backward_begin lr: 5.011872336272724e-07
[DEBUG]2020-06-20 05:28:40,784:utils:itr: 28, num_batch: 28, last loss: 1.000951, smooth_loss: 1.199151
[DEBUG]2020-06-20 05:28:40,794:utils:loss_avg: 1.19956, lr_pg0:5.011872336272724e-07, lr_pg1: 5.011872336272724e-07final_score:0.46440, mc_score:-0.02447
[DEBUG]2020-06-20 05:28:41,162:utils:grad info pg0: norm std(0.017333) mean(0.033880)
[DEBUG]2020-06-20 05:28:41,165:utils:grad info pg1: norm std(0.001440) mean(0.002769)
[DEBUG]2020-06-20 05:28:41,467:utils:on_backward_begin lr: 5.308844442309882e-07
[DEBUG]2020-06-20 05:28:41,469:utils:itr: 29, num_batch: 29, last loss: 1.104424, smooth_loss: 1.194983
[DEBUG]2020-06-20 05:28:41,478:utils:loss_avg: 1.19639, lr_pg0:5.308844442309882e-07, lr_pg1: 5.308844442309882e-07final_score:0.46001, mc_score:-0.01229
[DEBUG]2020-06-20 05:28:41,853:utils:grad info pg0: norm std(0.009510) mean(0.026434)
[DEBUG]2020-06-20 05:28:41,857:utils:grad info pg1: norm std(0.001333) mean(0.002532)
[DEBUG]2020-06-20 05:28:42,159:utils:on_backward_begin lr: 5.62341325190349e-07
[DEBUG]2020-06-20 05:28:42,161:utils:itr: 30, num_batch: 30, last loss: 1.088287, smooth_loss: 1.190398
[DEBUG]2020-06-20 05:28:42,171:utils:loss_avg: 1.19290, lr_pg0:5.62341325190349e-07, lr_pg1: 5.62341325190349e-07final_score:0.45804, mc_score:-0.00317
[DEBUG]2020-06-20 05:28:42,539:utils:grad info pg0: norm std(0.040024) mean(0.049449)
[DEBUG]2020-06-20 05:28:42,544:utils:grad info pg1: norm std(0.001509) mean(0.002827)
[DEBUG]2020-06-20 05:28:42,842:utils:on_backward_begin lr: 5.956621435290105e-07
[DEBUG]2020-06-20 05:28:42,844:utils:itr: 31, num_batch: 31, last loss: 1.027422, smooth_loss: 1.183552
[DEBUG]2020-06-20 05:28:42,857:utils:loss_avg: 1.18773, lr_pg0:5.956621435290105e-07, lr_pg1: 5.956621435290105e-07final_score:0.46380, mc_score:0.00755
[DEBUG]2020-06-20 05:28:43,225:utils:grad info pg0: norm std(0.007138) mean(0.021775)
[DEBUG]2020-06-20 05:28:43,229:utils:grad info pg1: norm std(0.001015) mean(0.001902)
[DEBUG]2020-06-20 05:28:43,528:utils:on_backward_begin lr: 6.309573444801932e-07
[DEBUG]2020-06-20 05:28:43,530:utils:itr: 32, num_batch: 32, last loss: 0.939182, smooth_loss: 1.173508

[DEBUG]2020-06-20 05:28:43,539:utils:loss_avg: 1.18020, lr_pg0:6.309573444801932e-07, lr_pg1: 6.309573444801932e-07final_score:0.46462, mc_score:0.01256
[DEBUG]2020-06-20 05:28:43,907:utils:grad info pg0: norm std(0.005150) mean(0.014589)
[DEBUG]2020-06-20 05:28:43,910:utils:grad info pg1: norm std(0.000769) mean(0.001417)
[DEBUG]2020-06-20 05:28:44,212:utils:on_backward_begin lr: 6.683439175686146e-07
[DEBUG]2020-06-20 05:28:44,213:utils:itr: 33, num_batch: 33, last loss: 1.094394, smooth_loss: 1.170323
[DEBUG]2020-06-20 05:28:44,225:utils:loss_avg: 1.17768, lr_pg0:6.683439175686146e-07, lr_pg1: 6.683439175686146e-07final_score:0.46571, mc_score:0.00118
[DEBUG]2020-06-20 05:28:44,594:utils:grad info pg0: norm std(0.008971) mean(0.025737)
[DEBUG]2020-06-20 05:28:44,597:utils:grad info pg1: norm std(0.001169) mean(0.002260)
[DEBUG]2020-06-20 05:28:44,901:utils:on_backward_begin lr: 7.07945784384138e-07
[DEBUG]2020-06-20 05:28:44,903:utils:itr: 34, num_batch: 34, last loss: 1.041066, smooth_loss: 1.165224
[DEBUG]2020-06-20 05:28:44,912:utils:loss_avg: 1.17377, lr_pg0:7.07945784384138e-07, lr_pg1: 7.07945784384138e-07final_score:0.47329, mc_score:0.01796
[DEBUG]2020-06-20 05:28:45,280:utils:grad info pg0: norm std(0.011138) mean(0.025985)
[DEBUG]2020-06-20 05:28:45,283:utils:grad info pg1: norm std(0.001631) mean(0.002718)
[DEBUG]2020-06-20 05:28:45,585:utils:on_backward_begin lr: 7.498942093324557e-07
[DEBUG]2020-06-20 05:28:45,587:utils:itr: 35, num_batch: 35, last loss: 1.145607, smooth_loss: 1.164465
[DEBUG]2020-06-20 05:28:45,597:utils:loss_avg: 1.17299, lr_pg0:7.498942093324557e-07, lr_pg1: 7.498942093324557e-07final_score:0.46363, mc_score:-0.00636
[DEBUG]2020-06-20 05:28:45,966:utils:grad info pg0: norm std(0.007302) mean(0.020839)
[DEBUG]2020-06-20 05:28:45,970:utils:grad info pg1: norm std(0.001359) mean(0.002324)
[DEBUG]2020-06-20 05:28:46,266:utils:on_backward_begin lr: 7.943282347242813e-07
[DEBUG]2020-06-20 05:28:46,267:utils:itr: 36, num_batch: 36, last loss: 0.960571, smooth_loss: 1.156719
[DEBUG]2020-06-20 05:28:46,278:utils:loss_avg: 1.16725, lr_pg0:7.943282347242813e-07, lr_pg1: 7.943282347242813e-07final_score:0.46726, mc_score:-0.01072
[DEBUG]2020-06-20 05:28:46,647:utils:grad info pg0: norm std(0.025377) mean(0.060858)
[DEBUG]2020-06-20 05:28:46,650:utils:grad info pg1: norm std(0.002154) mean(0.004404)
[DEBUG]2020-06-20 05:28:46,952:utils:on_backward_begin lr: 8.413951416451951e-07
[DEBUG]2020-06-20 05:28:46,954:utils:itr: 37, num_batch: 37, last loss: 1.132980, smooth_loss: 1.155833
[DEBUG]2020-06-20 05:28:46,965:utils:loss_avg: 1.16635, lr_pg0:8.413951416451951e-07, lr_pg1: 8.413951416451951e-07final_score:0.46619, mc_score:-0.01067
[DEBUG]2020-06-20 05:28:47,332:utils:grad info pg0: norm std(0.090734) mean(0.063926)
[DEBUG]2020-06-20 05:28:47,335:utils:grad info pg1: norm std(0.001339) mean(0.002620)
[DEBUG]2020-06-20 05:28:47,635:utils:on_backward_begin lr: 8.912509381337455e-07
[DEBUG]2020-06-20 05:28:47,636:utils:itr: 38, num_batch: 38, last loss: 0.805448, smooth_loss: 1.142980
[DEBUG]2020-06-20 05:28:47,645:utils:loss_avg: 1.15709, lr_pg0:8.912509381337455e-07, lr_pg1: 8.912509381337455e-07final_score:0.46328, mc_score:0.00021
[DEBUG]2020-06-20 05:28:48,019:utils:grad info pg0: norm std(0.006276) mean(0.014231)
[DEBUG]2020-06-20 05:28:48,023:utils:grad info pg1: norm std(0.000668) mean(0.001304)
[DEBUG]2020-06-20 05:28:48,331:utils:on_backward_begin lr: 9.440608762859234e-07
[DEBUG]2020-06-20 05:28:48,333:utils:itr: 39, num_batch: 39, last loss: 0.807610, smooth_loss: 1.130879
[DEBUG]2020-06-20 05:28:48,343:utils:loss_avg: 1.14836, lr_pg0:9.440608762859234e-07, lr_pg1: 9.440608762859234e-07final_score:0.46703, mc_score:0.01019
[DEBUG]2020-06-20 05:28:48,717:utils:grad info pg0: norm std(0.009246) mean(0.029380)
[DEBUG]2020-06-20 05:28:48,721:utils:grad info pg1: norm std(0.001436) mean(0.002699)
[DEBUG]2020-06-20 05:28:49,027:utils:on_backward_begin lr: 1.0000000000000002e-06
[DEBUG]2020-06-20 05:28:49,029:utils:itr: 40, num_batch: 40, last loss: 1.305859, smooth_loss: 1.137093
[DEBUG]2020-06-20 05:28:49,037:utils:loss_avg: 1.15220, lr_pg0:1.0000000000000002e-06, lr_pg1: 1.0000000000000002e-06final_score:0.45477, mc_score:-0.00263
[DEBUG]2020-06-20 05:28:49,404:utils:grad info pg0: norm std(0.011554) mean(0.031400)
[DEBUG]2020-06-20 05:28:49,407:utils:grad info pg1: norm std(0.001524) mean(0.002828)
[DEBUG]2020-06-20 05:28:49,705:utils:on_backward_begin lr: 1.0592537251772886e-06
[DEBUG]2020-06-20 05:28:49,706:utils:itr: 41, num_batch: 41, last loss: 0.940573, smooth_loss: 1.130221
[DEBUG]2020-06-20 05:28:49,716:utils:loss_avg: 1.14716, lr_pg0:1.0592537251772886e-06, lr_pg1: 1.0592537251772886e-06final_score:0.46041, mc_score:0.00512
[DEBUG]2020-06-20 05:28:50,086:utils:grad info pg0: norm std(0.012212) mean(0.032370)
[DEBUG]2020-06-20 05:28:50,089:utils:grad info pg1: norm std(0.001669) mean(0.002936)
[DEBUG]2020-06-20 05:28:50,390:utils:on_backward_begin lr: 1.1220184543019633e-06
[DEBUG]2020-06-20 05:28:50,391:utils:itr: 42, num_batch: 42, last loss: 1.103603, smooth_loss: 1.129304
[DEBUG]2020-06-20 05:28:50,401:utils:loss_avg: 1.14615, lr_pg0:1.1220184543019633e-06, lr_pg1: 1.1220184543019633e-06final_score:0.46182, mc_score:0.00487
[DEBUG]2020-06-20 05:28:50,769:utils:grad info pg0: norm std(0.007548) mean(0.024475)
[DEBUG]2020-06-20 05:28:50,772:utils:grad info pg1: norm std(0.001040) mean(0.001873)
[DEBUG]2020-06-20 05:28:51,078:utils:on_backward_begin lr: 1.1885022274370185e-06
[DEBUG]2020-06-20 05:28:51,080:utils:itr: 43, num_batch: 43, last loss: 0.705544, smooth_loss: 1.114912
[DEBUG]2020-06-20 05:28:51,090:utils:loss_avg: 1.13613, lr_pg0:1.1885022274370185e-06, lr_pg1: 1.1885022274370185e-06final_score:0.46905, mc_score:0.01345
[DEBUG]2020-06-20 05:28:51,458:utils:grad info pg0: norm std(0.008287) mean(0.022028)
[DEBUG]2020-06-20 05:28:51,462:utils:grad info pg1: norm std(0.000931) mean(0.001818)
[DEBUG]2020-06-20 05:28:51,765:utils:on_backward_begin lr: 1.2589254117941672e-06
[DEBUG]2020-06-20 05:28:51,767:utils:itr: 44, num_batch: 44, last loss: 1.261177, smooth_loss: 1.119811
[DEBUG]2020-06-20 05:28:51,775:utils:loss_avg: 1.13891, lr_pg0:1.2589254117941672e-06, lr_pg1: 1.2589254117941672e-06final_score:0.46903, mc_score:0.02177
[DEBUG]2020-06-20 05:28:52,148:utils:grad info pg0: norm std(0.009975) mean(0.029114)
[DEBUG]2020-06-20 05:28:52,153:utils:grad info pg1: norm std(0.001259) mean(0.002449)
[DEBUG]2020-06-20 05:28:52,461:utils:on_backward_begin lr: 1.333521432163324e-06
[DEBUG]2020-06-20 05:28:52,463:utils:itr: 45, num_batch: 45, last loss: 0.943713, smooth_loss: 1.113992
[DEBUG]2020-06-20 05:28:52,473:utils:loss_avg: 1.13467, lr_pg0:1.333521432163324e-06, lr_pg1: 1.333521432163324e-06final_score:0.46800, mc_score:0.03039
[DEBUG]2020-06-20 05:28:52,841:utils:grad info pg0: norm std(0.008459) mean(0.022953)
[DEBUG]2020-06-20 05:28:52,844:utils:grad info pg1: norm std(0.001006) mean(0.001946)
[DEBUG]2020-06-20 05:28:53,158:utils:on_backward_begin lr: 1.4125375446227544e-06
[DEBUG]2020-06-20 05:28:53,159:utils:itr: 46, num_batch: 46, last loss: 1.042393, smooth_loss: 1.111656
[DEBUG]2020-06-20 05:28:53,169:utils:loss_avg: 1.13271, lr_pg0:1.4125375446227544e-06, lr_pg1: 1.4125375446227544e-06final_score:0.46782, mc_score:0.02961
[DEBUG]2020-06-20 05:28:53,539:utils:grad info pg0: norm std(0.004045) mean(0.011150)
[DEBUG]2020-06-20 05:28:53,541:utils:grad info pg1: norm std(0.000596) mean(0.001079)
[DEBUG]2020-06-20 05:28:53,840:utils:on_backward_begin lr: 1.496235656094433e-06
[DEBUG]2020-06-20 05:28:53,842:utils:itr: 47, num_batch: 47, last loss: 1.127785, smooth_loss: 1.112176
[DEBUG]2020-06-20 05:28:53,851:utils:loss_avg: 1.13260, lr_pg0:1.496235656094433e-06, lr_pg1: 1.496235656094433e-06final_score:0.46836, mc_score:0.02629
[DEBUG]2020-06-20 05:28:54,226:utils:grad info pg0: norm std(0.006039) mean(0.013362)
[DEBUG]2020-06-20 05:28:54,229:utils:grad info pg1: norm std(0.000683) mean(0.001260)
[DEBUG]2020-06-20 05:28:54,541:utils:on_backward_begin lr: 1.5848931924611132e-06
[DEBUG]2020-06-20 05:28:54,543:utils:itr: 48, num_batch: 48, last loss: 1.157943, smooth_loss: 1.113632
[DEBUG]2020-06-20 05:28:54,553:utils:loss_avg: 1.13312, lr_pg0:1.5848931924611132e-06, lr_pg1: 1.5848931924611132e-06final_score:0.46730, mc_score:0.02249

[DEBUG]2020-06-20 05:28:54,922:utils:grad info pg0: norm std(0.009629) mean(0.027957)
[DEBUG]2020-06-20 05:28:54,925:utils:grad info pg1: norm std(0.001479) mean(0.002665)
[DEBUG]2020-06-20 05:28:55,221:utils:on_backward_begin lr: 1.6788040181225603e-06
[DEBUG]2020-06-20 05:28:55,222:utils:itr: 49, num_batch: 49, last loss: 0.995783, smooth_loss: 1.109925
[DEBUG]2020-06-20 05:28:55,233:utils:loss_avg: 1.13037, lr_pg0:1.6788040181225603e-06, lr_pg1: 1.6788040181225603e-06final_score:0.46654, mc_score:0.01851
[DEBUG]2020-06-20 05:28:55,601:utils:grad info pg0: norm std(0.005092) mean(0.016397)
[DEBUG]2020-06-20 05:28:55,604:utils:grad info pg1: norm std(0.000753) mean(0.001445)
[DEBUG]2020-06-20 05:28:55,915:utils:on_backward_begin lr: 1.7782794100389227e-06
[DEBUG]2020-06-20 05:28:55,917:utils:itr: 50, num_batch: 50, last loss: 1.147568, smooth_loss: 1.111096
[DEBUG]2020-06-20 05:28:55,928:utils:loss_avg: 1.13071, lr_pg0:1.7782794100389227e-06, lr_pg1: 1.7782794100389227e-06final_score:0.46481, mc_score:0.00553
[DEBUG]2020-06-20 05:28:56,295:utils:grad info pg0: norm std(0.082156) mean(0.070343)
[DEBUG]2020-06-20 05:28:56,299:utils:grad info pg1: norm std(0.002195) mean(0.003346)
[DEBUG]2020-06-20 05:28:56,609:utils:on_backward_begin lr: 1.8836490894898005e-06
[DEBUG]2020-06-20 05:28:56,611:utils:itr: 51, num_batch: 51, last loss: 1.017979, smooth_loss: 1.108232
[DEBUG]2020-06-20 05:28:56,620:utils:loss_avg: 1.12854, lr_pg0:1.8836490894898005e-06, lr_pg1: 1.8836490894898005e-06final_score:0.46307, mc_score:0.00181
[DEBUG]2020-06-20 05:28:56,989:utils:grad info pg0: norm std(0.010179) mean(0.026564)
[DEBUG]2020-06-20 05:28:56,992:utils:grad info pg1: norm std(0.001179) mean(0.002333)
[DEBUG]2020-06-20 05:28:57,318:utils:on_backward_begin lr: 1.9952623149688796e-06
[DEBUG]2020-06-20 05:28:57,320:utils:itr: 52, num_batch: 52, last loss: 0.936563, smooth_loss: 1.103008
[DEBUG]2020-06-20 05:28:57,331:utils:loss_avg: 1.12492, lr_pg0:1.9952623149688796e-06, lr_pg1: 1.9952623149688796e-06final_score:0.46454, mc_score:0.00419
[DEBUG]2020-06-20 05:28:57,699:utils:grad info pg0: norm std(0.007732) mean(0.022311)
[DEBUG]2020-06-20 05:28:57,703:utils:grad info pg1: norm std(0.000882) mean(0.001773)
[DEBUG]2020-06-20 05:28:58,001:utils:on_backward_begin lr: 2.113489039836647e-06
[DEBUG]2020-06-20 05:28:58,002:utils:itr: 53, num_batch: 53, last loss: 0.848185, smooth_loss: 1.095334
[DEBUG]2020-06-20 05:28:58,011:utils:loss_avg: 1.11980, lr_pg0:2.113489039836647e-06, lr_pg1: 2.113489039836647e-06final_score:0.46668, mc_score:0.00890
[DEBUG]2020-06-20 05:28:58,382:utils:grad info pg0: norm std(0.019857) mean(0.039674)
[DEBUG]2020-06-20 05:28:58,386:utils:grad info pg1: norm std(0.002134) mean(0.003945)
[DEBUG]2020-06-20 05:28:58,687:utils:on_backward_begin lr: 2.23872113856834e-06
[DEBUG]2020-06-20 05:28:58,688:utils:itr: 54, num_batch: 54, last loss: 0.908940, smooth_loss: 1.089777
[DEBUG]2020-06-20 05:28:58,698:utils:loss_avg: 1.11596, lr_pg0:2.23872113856834e-06, lr_pg1: 2.23872113856834e-06final_score:0.46607, mc_score:0.01118
[DEBUG]2020-06-20 05:28:59,068:utils:grad info pg0: norm std(0.008245) mean(0.023828)
[DEBUG]2020-06-20 05:28:59,071:utils:grad info pg1: norm std(0.001067) mean(0.002064)
[DEBUG]2020-06-20 05:28:59,372:utils:on_backward_begin lr: 2.3713737056616556e-06
[DEBUG]2020-06-20 05:28:59,373:utils:itr: 55, num_batch: 55, last loss: 1.001200, smooth_loss: 1.087162
[DEBUG]2020-06-20 05:28:59,382:utils:loss_avg: 1.11391, lr_pg0:2.3713737056616556e-06, lr_pg1: 2.3713737056616556e-06final_score:0.46840, mc_score:0.00233
[DEBUG]2020-06-20 05:28:59,749:utils:grad info pg0: norm std(0.011771) mean(0.034092)
[DEBUG]2020-06-20 05:28:59,752:utils:grad info pg1: norm std(0.001465) mean(0.002761)
[DEBUG]2020-06-20 05:29:00,066:utils:on_backward_begin lr: 2.511886431509581e-06
[DEBUG]2020-06-20 05:29:00,067:utils:itr: 56, num_batch: 56, last loss: 0.997324, smooth_loss: 1.084534
[DEBUG]2020-06-20 05:29:00,078:utils:loss_avg: 1.11187, lr_pg0:2.511886431509581e-06, lr_pg1: 2.511886431509581e-06final_score:0.47008, mc_score:0.00229
[DEBUG]2020-06-20 05:29:00,448:utils:grad info pg0: norm std(0.003498) mean(0.010627)
[DEBUG]2020-06-20 05:29:00,451:utils:grad info pg1: norm std(0.000539) mean(0.000994)
[DEBUG]2020-06-20 05:29:00,749:utils:on_backward_begin lr: 2.6607250597988086e-06
[DEBUG]2020-06-20 05:29:00,751:utils:itr: 57, num_batch: 57, last loss: 0.933422, smooth_loss: 1.080155
[DEBUG]2020-06-20 05:29:00,761:utils:loss_avg: 1.10879, lr_pg0:2.6607250597988086e-06, lr_pg1: 2.6607250597988086e-06final_score:0.47572, mc_score:0.01567
[DEBUG]2020-06-20 05:29:01,129:utils:grad info pg0: norm std(0.010984) mean(0.036549)
[DEBUG]2020-06-20 05:29:01,133:utils:grad info pg1: norm std(0.001539) mean(0.002950)
[DEBUG]2020-06-20 05:29:01,430:utils:on_backward_begin lr: 2.818382931264453e-06
[DEBUG]2020-06-20 05:29:01,431:utils:itr: 58, num_batch: 58, last loss: 0.929185, smooth_loss: 1.075819
[DEBUG]2020-06-20 05:29:01,441:utils:loss_avg: 1.10575, lr_pg0:2.818382931264453e-06, lr_pg1: 2.818382931264453e-06final_score:0.47999, mc_score:0.01803
[DEBUG]2020-06-20 05:29:01,810:utils:grad info pg0: norm std(0.007743) mean(0.025288)
[DEBUG]2020-06-20 05:29:01,813:utils:grad info pg1: norm std(0.001027) mean(0.001940)
[DEBUG]2020-06-20 05:29:02,113:utils:on_backward_begin lr: 2.985382618917959e-06
[DEBUG]2020-06-20 05:29:02,115:utils:itr: 59, num_batch: 59, last loss: 0.926494, smooth_loss: 1.071568
[DEBUG]2020-06-20 05:29:02,124:utils:loss_avg: 1.10276, lr_pg0:2.985382618917959e-06, lr_pg1: 2.985382618917959e-06final_score:0.48664, mc_score:0.02773
[DEBUG]2020-06-20 05:29:02,491:utils:grad info pg0: norm std(0.005437) mean(0.016130)
[DEBUG]2020-06-20 05:29:02,494:utils:grad info pg1: norm std(0.000800) mean(0.001471)
[DEBUG]2020-06-20 05:29:02,791:utils:on_backward_begin lr: 3.1622776601683788e-06
[DEBUG]2020-06-20 05:29:02,793:utils:itr: 60, num_batch: 60, last loss: 1.101717, smooth_loss: 1.072419
[DEBUG]2020-06-20 05:29:02,803:utils:loss_avg: 1.10274, lr_pg0:3.1622776601683788e-06, lr_pg1: 3.1622776601683788e-06final_score:0.48133, mc_score:0.02204
[DEBUG]2020-06-20 05:29:03,171:utils:grad info pg0: norm std(0.009394) mean(0.027456)
[DEBUG]2020-06-20 05:29:03,174:utils:grad info pg1: norm std(0.001240) mean(0.002380)
[DEBUG]2020-06-20 05:29:03,473:utils:on_backward_begin lr: 3.3496543915782763e-06
[DEBUG]2020-06-20 05:29:03,474:utils:itr: 61, num_batch: 61, last loss: 0.865856, smooth_loss: 1.066635
[DEBUG]2020-06-20 05:29:03,484:utils:loss_avg: 1.09892, lr_pg0:3.3496543915782763e-06, lr_pg1: 3.3496543915782763e-06final_score:0.48789, mc_score:0.03142
[DEBUG]2020-06-20 05:29:03,852:utils:grad info pg0: norm std(0.010075) mean(0.023569)
[DEBUG]2020-06-20 05:29:03,856:utils:grad info pg1: norm std(0.000969) mean(0.001929)
[DEBUG]2020-06-20 05:29:04,162:utils:on_backward_begin lr: 3.5481338923357546e-06
[DEBUG]2020-06-20 05:29:04,164:utils:itr: 62, num_batch: 62, last loss: 1.032806, smooth_loss: 1.065695
[DEBUG]2020-06-20 05:29:04,174:utils:loss_avg: 1.09787, lr_pg0:3.5481338923357546e-06, lr_pg1: 3.5481338923357546e-06final_score:0.48890, mc_score:0.02417
[DEBUG]2020-06-20 05:29:04,541:utils:grad info pg0: norm std(0.012163) mean(0.014261)
[DEBUG]2020-06-20 05:29:04,544:utils:grad info pg1: norm std(0.000512) mean(0.000941)
[DEBUG]2020-06-20 05:29:04,858:utils:on_backward_begin lr: 3.7583740428844413e-06
[DEBUG]2020-06-20 05:29:04,860:utils:itr: 63, num_batch: 63, last loss: 0.995863, smooth_loss: 1.063770
[DEBUG]2020-06-20 05:29:04,870:utils:loss_avg: 1.09628, lr_pg0:3.7583740428844413e-06, lr_pg1: 3.7583740428844413e-06final_score:0.48685, mc_score:0.01864
[DEBUG]2020-06-20 05:29:05,238:utils:grad info pg0: norm std(0.004055) mean(0.011154)
[DEBUG]2020-06-20 05:29:05,241:utils:grad info pg1: norm std(0.000528) mean(0.001003)
[DEBUG]2020-06-20 05:29:05,544:utils:on_backward_begin lr: 3.9810717055349725e-06
[DEBUG]2020-06-20 05:29:05,546:utils:itr: 64, num_batch: 64, last loss: 1.127073, smooth_loss: 1.065502
[DEBUG]2020-06-20 05:29:05,556:utils:loss_avg: 1.09675, lr_pg0:3.9810717055349725e-06, lr_pg1: 3.9810717055349725e-06final_score:0.48234, mc_score:0.00448
[DEBUG]2020-06-20 05:29:05,923:utils:grad info pg0: norm std(0.005964) mean(0.015514)

[DEBUG]2020-06-20 05:29:05,927:utils:grad info pg1: norm std(0.000889) mean(0.001581)
[DEBUG]2020-06-20 05:29:06,226:utils:on_backward_begin lr: 4.216965034285823e-06
[DEBUG]2020-06-20 05:29:06,228:utils:itr: 65, num_batch: 65, last loss: 0.922428, smooth_loss: 1.061616
[DEBUG]2020-06-20 05:29:06,237:utils:loss_avg: 1.09411, lr_pg0:4.216965034285823e-06, lr_pg1: 4.216965034285823e-06final_score:0.48514, mc_score:0.00875
[DEBUG]2020-06-20 05:29:06,605:utils:grad info pg0: norm std(0.008722) mean(0.021787)
[DEBUG]2020-06-20 05:29:06,609:utils:grad info pg1: norm std(0.000969) mean(0.001885)
[DEBUG]2020-06-20 05:29:06,917:utils:on_backward_begin lr: 4.466835921509631e-06
[DEBUG]2020-06-20 05:29:06,919:utils:itr: 66, num_batch: 66, last loss: 0.926322, smooth_loss: 1.057968
[DEBUG]2020-06-20 05:29:06,928:utils:loss_avg: 1.09161, lr_pg0:4.466835921509631e-06, lr_pg1: 4.466835921509631e-06final_score:0.48344, mc_score:0.00863
[DEBUG]2020-06-20 05:29:07,297:utils:grad info pg0: norm std(0.011074) mean(0.031426)
[DEBUG]2020-06-20 05:29:07,300:utils:grad info pg1: norm std(0.001614) mean(0.003040)
[DEBUG]2020-06-20 05:29:07,600:utils:on_backward_begin lr: 4.7315125896148055e-06
[DEBUG]2020-06-20 05:29:07,601:utils:itr: 67, num_batch: 67, last loss: 1.051260, smooth_loss: 1.057788
[DEBUG]2020-06-20 05:29:07,611:utils:loss_avg: 1.09101, lr_pg0:4.7315125896148055e-06, lr_pg1: 4.7315125896148055e-06final_score:0.48505, mc_score:0.01259
[DEBUG]2020-06-20 05:29:07,979:utils:grad info pg0: norm std(0.007279) mean(0.022201)
[DEBUG]2020-06-20 05:29:07,983:utils:grad info pg1: norm std(0.001161) mean(0.002084)
[DEBUG]2020-06-20 05:29:08,316:utils:on_backward_begin lr: 5.011872336272724e-06
[DEBUG]2020-06-20 05:29:08,317:utils:itr: 68, num_batch: 68, last loss: 1.079937, smooth_loss: 1.058377
[DEBUG]2020-06-20 05:29:08,327:utils:loss_avg: 1.09085, lr_pg0:5.011872336272724e-06, lr_pg1: 5.011872336272724e-06final_score:0.48536, mc_score:0.01019
[DEBUG]2020-06-20 05:29:08,695:utils:grad info pg0: norm std(0.003668) mean(0.011986)
[DEBUG]2020-06-20 05:29:08,699:utils:grad info pg1: norm std(0.000566) mean(0.001094)
[DEBUG]2020-06-20 05:29:09,001:utils:on_backward_begin lr: 5.308844442309882e-06
[DEBUG]2020-06-20 05:29:09,002:utils:itr: 69, num_batch: 69, last loss: 1.171705, smooth_loss: 1.061372
[DEBUG]2020-06-20 05:29:09,013:utils:loss_avg: 1.09201, lr_pg0:5.308844442309882e-06, lr_pg1: 5.308844442309882e-06final_score:0.48590, mc_score:0.00773
[DEBUG]2020-06-20 05:29:09,382:utils:grad info pg0: norm std(0.010765) mean(0.025469)
[DEBUG]2020-06-20 05:29:09,386:utils:grad info pg1: norm std(0.001531) mean(0.002645)
[DEBUG]2020-06-20 05:29:09,691:utils:on_backward_begin lr: 5.6234132519034895e-06
[DEBUG]2020-06-20 05:29:09,693:utils:itr: 70, num_batch: 70, last loss: 0.903222, smooth_loss: 1.057220
[DEBUG]2020-06-20 05:29:09,701:utils:loss_avg: 1.08935, lr_pg0:5.6234132519034895e-06, lr_pg1: 5.6234132519034895e-06final_score:0.48439, mc_score:0.00000
[DEBUG]2020-06-20 05:29:10,070:utils:grad info pg0: norm std(0.017063) mean(0.039708)
[DEBUG]2020-06-20 05:29:10,073:utils:grad info pg1: norm std(0.001692) mean(0.003248)
[DEBUG]2020-06-20 05:29:10,372:utils:on_backward_begin lr: 5.9566214352901035e-06
[DEBUG]2020-06-20 05:29:10,374:utils:itr: 71, num_batch: 71, last loss: 0.810684, smooth_loss: 1.050787
[DEBUG]2020-06-20 05:29:10,383:utils:loss_avg: 1.08548, lr_pg0:5.9566214352901035e-06, lr_pg1: 5.9566214352901035e-06final_score:0.48925, mc_score:0.00938
[DEBUG]2020-06-20 05:29:10,753:utils:grad info pg0: norm std(0.009699) mean(0.025916)
[DEBUG]2020-06-20 05:29:10,756:utils:grad info pg1: norm std(0.001164) mean(0.002108)
[DEBUG]2020-06-20 05:29:11,061:utils:on_backward_begin lr: 6.309573444801931e-06
[DEBUG]2020-06-20 05:29:11,063:utils:itr: 72, num_batch: 72, last loss: 1.079460, smooth_loss: 1.051531
[DEBUG]2020-06-20 05:29:11,072:utils:loss_avg: 1.08540, lr_pg0:6.309573444801931e-06, lr_pg1: 6.309573444801931e-06final_score:0.48459, mc_score:0.00812
[DEBUG]2020-06-20 05:29:11,443:utils:grad info pg0: norm std(0.009475) mean(0.023101)
[DEBUG]2020-06-20 05:29:11,446:utils:grad info pg1: norm std(0.001144) mean(0.002097)
[DEBUG]2020-06-20 05:29:11,746:utils:on_backward_begin lr: 6.683439175686145e-06
[DEBUG]2020-06-20 05:29:11,748:utils:itr: 73, num_batch: 73, last loss: 1.077564, smooth_loss: 1.052202
[DEBUG]2020-06-20 05:29:11,759:utils:loss_avg: 1.08529, lr_pg0:6.683439175686145e-06, lr_pg1: 6.683439175686145e-06final_score:0.48546, mc_score:0.00747
[DEBUG]2020-06-20 05:29:12,128:utils:grad info pg0: norm std(0.011062) mean(0.027645)
[DEBUG]2020-06-20 05:29:12,131:utils:grad info pg1: norm std(0.001261) mean(0.002420)
[DEBUG]2020-06-20 05:29:12,428:utils:on_backward_begin lr: 7.079457843841379e-06
[DEBUG]2020-06-20 05:29:12,430:utils:itr: 74, num_batch: 74, last loss: 0.963104, smooth_loss: 1.049918
[DEBUG]2020-06-20 05:29:12,440:utils:loss_avg: 1.08366, lr_pg0:7.079457843841379e-06, lr_pg1: 7.079457843841379e-06final_score:0.48627, mc_score:0.00737
[DEBUG]2020-06-20 05:29:12,808:utils:grad info pg0: norm std(0.011621) mean(0.035254)
[DEBUG]2020-06-20 05:29:12,812:utils:grad info pg1: norm std(0.001608) mean(0.003241)
[DEBUG]2020-06-20 05:29:13,113:utils:on_backward_begin lr: 7.498942093324558e-06
[DEBUG]2020-06-20 05:29:13,115:utils:itr: 75, num_batch: 75, last loss: 0.924305, smooth_loss: 1.046716
[DEBUG]2020-06-20 05:29:13,124:utils:loss_avg: 1.08156, lr_pg0:7.498942093324558e-06, lr_pg1: 7.498942093324558e-06final_score:0.48608, mc_score:0.00782
[DEBUG]2020-06-20 05:29:13,502:utils:grad info pg0: norm std(0.004703) mean(0.014252)
[DEBUG]2020-06-20 05:29:13,505:utils:grad info pg1: norm std(0.000716) mean(0.001337)
[DEBUG]2020-06-20 05:29:13,802:utils:on_backward_begin lr: 7.943282347242815e-06
[DEBUG]2020-06-20 05:29:13,804:utils:itr: 76, num_batch: 76, last loss: 0.819288, smooth_loss: 1.040951
[DEBUG]2020-06-20 05:29:13,813:utils:loss_avg: 1.07816, lr_pg0:7.943282347242815e-06, lr_pg1: 7.943282347242815e-06final_score:0.48889, mc_score:0.01131
[DEBUG]2020-06-20 05:29:14,180:utils:grad info pg0: norm std(0.006420) mean(0.018438)
[DEBUG]2020-06-20 05:29:14,184:utils:grad info pg1: norm std(0.000827) mean(0.001614)
[DEBUG]2020-06-20 05:29:14,481:utils:on_backward_begin lr: 8.413951416451952e-06
[DEBUG]2020-06-20 05:29:14,483:utils:itr: 77, num_batch: 77, last loss: 1.055803, smooth_loss: 1.041325
[DEBUG]2020-06-20 05:29:14,494:utils:loss_avg: 1.07787, lr_pg0:8.413951416451952e-06, lr_pg1: 8.413951416451952e-06final_score:0.48694, mc_score:0.00763
[DEBUG]2020-06-20 05:29:14,862:utils:grad info pg0: norm std(0.007655) mean(0.025382)
[DEBUG]2020-06-20 05:29:14,865:utils:grad info pg1: norm std(0.001293) mean(0.002378)
[DEBUG]2020-06-20 05:29:15,167:utils:on_backward_begin lr: 8.912509381337456e-06
[DEBUG]2020-06-20 05:29:15,169:utils:itr: 78, num_batch: 78, last loss: 0.907665, smooth_loss: 1.037973
[DEBUG]2020-06-20 05:29:15,178:utils:loss_avg: 1.07572, lr_pg0:8.912509381337456e-06, lr_pg1: 8.912509381337456e-06final_score:0.48899, mc_score:0.01243
[DEBUG]2020-06-20 05:29:15,546:utils:grad info pg0: norm std(0.003828) mean(0.011846)
[DEBUG]2020-06-20 05:29:15,549:utils:grad info pg1: norm std(0.000610) mean(0.001150)
[DEBUG]2020-06-20 05:29:15,849:utils:on_backward_begin lr: 9.440608762859235e-06
[DEBUG]2020-06-20 05:29:15,851:utils:itr: 79, num_batch: 79, last loss: 0.946598, smooth_loss: 1.035692
[DEBUG]2020-06-20 05:29:15,860:utils:loss_avg: 1.07410, lr_pg0:9.440608762859235e-06, lr_pg1: 9.440608762859235e-06final_score:0.49018, mc_score:0.01360
[DEBUG]2020-06-20 05:29:16,229:utils:grad info pg0: norm std(0.004853) mean(0.012724)
[DEBUG]2020-06-20 05:29:16,232:utils:grad info pg1: norm std(0.000636) mean(0.001195)
[DEBUG]2020-06-20 05:29:16,551:utils:on_backward_begin lr: 1.0000000000000003e-05
[DEBUG]2020-06-20 05:29:16,553:utils:itr: 80, num_batch: 80, last loss: 0.875893, smooth_loss: 1.031724
[DEBUG]2020-06-20 05:29:16,563:utils:loss_avg: 1.07166, lr_pg0:1.0000000000000003e-05, lr_pg1: 1.0000000000000003e-05final_score:0.48943, mc_score:0.01414
[DEBUG]2020-06-20 05:29:16,934:utils:grad info pg0: norm std(0.011364) mean(0.027829)
[DEBUG]2020-06-20 05:29:16,938:utils:grad info pg1: norm std(0.001496) mean(0.002790)

[DEBUG]2020-06-20 05:29:17,245:utils:on_backward_begin lr: 1.0592537251772892e-05
[DEBUG]2020-06-20 05:29:17,246:utils:itr: 81, num_batch: 81, last loss: 1.131146, smooth_loss: 1.034181
[DEBUG]2020-06-20 05:29:17,255:utils:loss_avg: 1.07238, lr_pg0:1.0592537251772892e-05, lr_pg1: 1.0592537251772892e-05final_score:0.48826, mc_score:0.01197
[DEBUG]2020-06-20 05:29:17,623:utils:grad info pg0: norm std(0.003794) mean(0.011139)
[DEBUG]2020-06-20 05:29:17,626:utils:grad info pg1: norm std(0.000619) mean(0.001111)
[DEBUG]2020-06-20 05:29:17,926:utils:on_backward_begin lr: 1.1220184543019632e-05
[DEBUG]2020-06-20 05:29:17,928:utils:itr: 82, num_batch: 82, last loss: 0.896249, smooth_loss: 1.030788
[DEBUG]2020-06-20 05:29:17,938:utils:loss_avg: 1.07026, lr_pg0:1.1220184543019632e-05, lr_pg1: 1.1220184543019632e-05final_score:0.49062, mc_score:0.01644
[DEBUG]2020-06-20 05:29:18,306:utils:grad info pg0: norm std(0.003900) mean(0.011818)
[DEBUG]2020-06-20 05:29:18,309:utils:grad info pg1: norm std(0.000653) mean(0.001187)
[DEBUG]2020-06-20 05:29:18,608:utils:on_backward_begin lr: 1.1885022274370181e-05
[DEBUG]2020-06-20 05:29:18,610:utils:itr: 83, num_batch: 83, last loss: 0.743115, smooth_loss: 1.023744
[DEBUG]2020-06-20 05:29:18,619:utils:loss_avg: 1.06636, lr_pg0:1.1885022274370181e-05, lr_pg1: 1.1885022274370181e-05final_score:0.49558, mc_score:0.02712
[DEBUG]2020-06-20 05:29:18,986:utils:grad info pg0: norm std(0.009402) mean(0.031112)
[DEBUG]2020-06-20 05:29:18,989:utils:grad info pg1: norm std(0.001289) mean(0.002554)
[DEBUG]2020-06-20 05:29:19,292:utils:on_backward_begin lr: 1.258925411794167e-05
[DEBUG]2020-06-20 05:29:19,294:utils:itr: 84, num_batch: 84, last loss: 0.944291, smooth_loss: 1.021807
[DEBUG]2020-06-20 05:29:19,304:utils:loss_avg: 1.06493, lr_pg0:1.258925411794167e-05, lr_pg1: 1.258925411794167e-05final_score:0.49349, mc_score:0.02432
[DEBUG]2020-06-20 05:29:19,671:utils:grad info pg0: norm std(0.003489) mean(0.010852)
[DEBUG]2020-06-20 05:29:19,675:utils:grad info pg1: norm std(0.000490) mean(0.000915)
[DEBUG]2020-06-20 05:29:19,971:utils:on_backward_begin lr: 1.3335214321633239e-05
[DEBUG]2020-06-20 05:29:19,973:utils:itr: 85, num_batch: 85, last loss: 0.940727, smooth_loss: 1.019839
[DEBUG]2020-06-20 05:29:19,982:utils:loss_avg: 1.06348, lr_pg0:1.3335214321633239e-05, lr_pg1: 1.3335214321633239e-05final_score:0.49564, mc_score:0.02829
[DEBUG]2020-06-20 05:29:20,349:utils:grad info pg0: norm std(0.014611) mean(0.036311)
[DEBUG]2020-06-20 05:29:20,352:utils:grad info pg1: norm std(0.001322) mean(0.002700)
[DEBUG]2020-06-20 05:29:20,649:utils:on_backward_begin lr: 1.4125375446227541e-05
[DEBUG]2020-06-20 05:29:20,651:utils:itr: 86, num_batch: 86, last loss: 0.940480, smooth_loss: 1.017921
[DEBUG]2020-06-20 05:29:20,659:utils:loss_avg: 1.06207, lr_pg0:1.4125375446227541e-05, lr_pg1: 1.4125375446227541e-05final_score:0.49838, mc_score:0.03014
[DEBUG]2020-06-20 05:29:21,026:utils:grad info pg0: norm std(0.006368) mean(0.017315)
[DEBUG]2020-06-20 05:29:21,029:utils:grad info pg1: norm std(0.000843) mean(0.001556)
[DEBUG]2020-06-20 05:29:21,329:utils:on_backward_begin lr: 1.4962356560944334e-05
[DEBUG]2020-06-20 05:29:21,330:utils:itr: 87, num_batch: 87, last loss: 1.137777, smooth_loss: 1.020806
[DEBUG]2020-06-20 05:29:21,339:utils:loss_avg: 1.06293, lr_pg0:1.4962356560944334e-05, lr_pg1: 1.4962356560944334e-05final_score:0.49617, mc_score:0.02352
[DEBUG]2020-06-20 05:29:21,713:utils:grad info pg0: norm std(0.004149) mean(0.009481)
[DEBUG]2020-06-20 05:29:21,717:utils:grad info pg1: norm std(0.000477) mean(0.000867)
[DEBUG]2020-06-20 05:29:22,015:utils:on_backward_begin lr: 1.5848931924611134e-05
[DEBUG]2020-06-20 05:29:22,016:utils:itr: 88, num_batch: 88, last loss: 0.808468, smooth_loss: 1.015716
[DEBUG]2020-06-20 05:29:22,025:utils:loss_avg: 1.06007, lr_pg0:1.5848931924611134e-05, lr_pg1: 1.5848931924611134e-05final_score:0.49880, mc_score:0.03034
[DEBUG]2020-06-20 05:29:22,393:utils:grad info pg0: norm std(0.013755) mean(0.034331)
[DEBUG]2020-06-20 05:29:22,396:utils:grad info pg1: norm std(0.001934) mean(0.003390)
[DEBUG]2020-06-20 05:29:22,702:utils:on_backward_begin lr: 1.6788040181225605e-05
[DEBUG]2020-06-20 05:29:22,704:utils:itr: 89, num_batch: 89, last loss: 0.944862, smooth_loss: 1.014025
[DEBUG]2020-06-20 05:29:22,713:utils:loss_avg: 1.05879, lr_pg0:1.6788040181225605e-05, lr_pg1: 1.6788040181225605e-05final_score:0.49972, mc_score:0.02980
[DEBUG]2020-06-20 05:29:23,081:utils:grad info pg0: norm std(0.011394) mean(0.033136)
[DEBUG]2020-06-20 05:29:23,085:utils:grad info pg1: norm std(0.001463) mean(0.002801)
[DEBUG]2020-06-20 05:29:23,382:utils:on_backward_begin lr: 1.778279410038923e-05
[DEBUG]2020-06-20 05:29:23,384:utils:itr: 90, num_batch: 90, last loss: 0.996590, smooth_loss: 1.013610
[DEBUG]2020-06-20 05:29:23,394:utils:loss_avg: 1.05811, lr_pg0:1.778279410038923e-05, lr_pg1: 1.778279410038923e-05final_score:0.49977, mc_score:0.02918
[DEBUG]2020-06-20 05:29:23,762:utils:grad info pg0: norm std(0.030230) mean(0.032606)
[DEBUG]2020-06-20 05:29:23,765:utils:grad info pg1: norm std(0.001017) mean(0.001879)
[DEBUG]2020-06-20 05:29:24,081:utils:on_backward_begin lr: 1.8836490894898008e-05
[DEBUG]2020-06-20 05:29:24,082:utils:itr: 91, num_batch: 91, last loss: 0.860654, smooth_loss: 1.009986
[DEBUG]2020-06-20 05:29:24,092:utils:loss_avg: 1.05596, lr_pg0:1.8836490894898008e-05, lr_pg1: 1.8836490894898008e-05final_score:0.50174, mc_score:0.03443
[DEBUG]2020-06-20 05:29:24,459:utils:grad info pg0: norm std(0.007283) mean(0.020822)
[DEBUG]2020-06-20 05:29:24,462:utils:grad info pg1: norm std(0.000941) mean(0.001769)
[DEBUG]2020-06-20 05:29:24,760:utils:on_backward_begin lr: 1.99526231496888e-05
[DEBUG]2020-06-20 05:29:24,762:utils:itr: 92, num_batch: 92, last loss: 0.779450, smooth_loss: 1.004544
[DEBUG]2020-06-20 05:29:24,772:utils:loss_avg: 1.05299, lr_pg0:1.99526231496888e-05, lr_pg1: 1.99526231496888e-05final_score:0.50458, mc_score:0.03761
[DEBUG]2020-06-20 05:29:25,139:utils:grad info pg0: norm std(0.008258) mean(0.018577)
[DEBUG]2020-06-20 05:29:25,142:utils:grad info pg1: norm std(0.000969) mean(0.001797)
[DEBUG]2020-06-20 05:29:25,452:utils:on_backward_begin lr: 2.1134890398366472e-05
[DEBUG]2020-06-20 05:29:25,454:utils:itr: 93, num_batch: 93, last loss: 1.022699, smooth_loss: 1.004971
[DEBUG]2020-06-20 05:29:25,465:utils:loss_avg: 1.05267, lr_pg0:2.1134890398366472e-05, lr_pg1: 2.1134890398366472e-05final_score:0.50481, mc_score:0.03700
[DEBUG]2020-06-20 05:29:25,833:utils:grad info pg0: norm std(0.010710) mean(0.027916)
[DEBUG]2020-06-20 05:29:25,836:utils:grad info pg1: norm std(0.001252) mean(0.002324)
[DEBUG]2020-06-20 05:29:26,142:utils:on_backward_begin lr: 2.238721138568339e-05
[DEBUG]2020-06-20 05:29:26,143:utils:itr: 94, num_batch: 94, last loss: 1.054593, smooth_loss: 1.006134
[DEBUG]2020-06-20 05:29:26,152:utils:loss_avg: 1.05269, lr_pg0:2.238721138568339e-05, lr_pg1: 2.238721138568339e-05final_score:0.50490, mc_score:0.03475
[DEBUG]2020-06-20 05:29:26,520:utils:grad info pg0: norm std(0.004921) mean(0.014139)
[DEBUG]2020-06-20 05:29:26,523:utils:grad info pg1: norm std(0.000763) mean(0.001375)
[DEBUG]2020-06-20 05:29:26,818:utils:on_backward_begin lr: 2.3713737056616544e-05
[DEBUG]2020-06-20 05:29:26,820:utils:itr: 95, num_batch: 95, last loss: 0.853186, smooth_loss: 1.002562
[DEBUG]2020-06-20 05:29:26,829:utils:loss_avg: 1.05061, lr_pg0:2.3713737056616544e-05, lr_pg1: 2.3713737056616544e-05final_score:0.50774, mc_score:0.03882
[DEBUG]2020-06-20 05:29:27,196:utils:grad info pg0: norm std(0.005294) mean(0.016919)
[DEBUG]2020-06-20 05:29:27,200:utils:grad info pg1: norm std(0.000741) mean(0.001383)
[DEBUG]2020-06-20 05:29:27,498:utils:on_backward_begin lr: 2.5118864315095795e-05
[DEBUG]2020-06-20 05:29:27,499:utils:itr: 96, num_batch: 96, last loss: 0.762915, smooth_loss: 0.996983
[DEBUG]2020-06-20 05:29:27,509:utils:loss_avg: 1.04764, lr_pg0:2.5118864315095795e-05, lr_pg1: 2.5118864315095795e-05final_score:0.51062, mc_score:0.03906
[DEBUG]2020-06-20 05:29:27,878:utils:grad info pg0: norm std(0.008328) mean(0.029911)
[DEBUG]2020-06-20 05:29:27,882:utils:grad info pg1: norm std(0.001577) mean(0.002819)

[DEBUG]2020-06-20 05:29:28,181:utils:on_backward_begin lr: 2.660725059798809e-05
[DEBUG]2020-06-20 05:29:28,182:utils:itr: 97, num_batch: 97, last loss: 0.796255, smooth_loss: 0.992325
[DEBUG]2020-06-20 05:29:28,191:utils:loss_avg: 1.04508, lr_pg0:2.660725059798809e-05, lr_pg1: 2.660725059798809e-05final_score:0.51350, mc_score:0.04165
[DEBUG]2020-06-20 05:29:28,559:utils:grad info pg0: norm std(0.003663) mean(0.010791)
[DEBUG]2020-06-20 05:29:28,563:utils:grad info pg1: norm std(0.000578) mean(0.001047)
[DEBUG]2020-06-20 05:29:28,867:utils:on_backward_begin lr: 2.8183829312644535e-05
[DEBUG]2020-06-20 05:29:28,868:utils:itr: 98, num_batch: 98, last loss: 0.996902, smooth_loss: 0.992431
[DEBUG]2020-06-20 05:29:28,879:utils:loss_avg: 1.04459, lr_pg0:2.8183829312644535e-05, lr_pg1: 2.8183829312644535e-05final_score:0.51232, mc_score:0.03772
[DEBUG]2020-06-20 05:29:29,247:utils:grad info pg0: norm std(0.011602) mean(0.027296)
[DEBUG]2020-06-20 05:29:29,250:utils:grad info pg1: norm std(0.001424) mean(0.002714)
[DEBUG]2020-06-20 05:29:29,557:utils:on_backward_begin lr: 2.9853826189179594e-05
[DEBUG]2020-06-20 05:29:29,558:utils:itr: 99, num_batch: 99, last loss: 0.788698, smooth_loss: 0.987733
[DEBUG]2020-06-20 05:29:29,569:utils:loss_avg: 1.04203, lr_pg0:2.9853826189179594e-05, lr_pg1: 2.9853826189179594e-05final_score:0.51318, mc_score:0.03896
[DEBUG]2020-06-20 05:29:29,936:utils:grad info pg0: norm std(0.005593) mean(0.017566)
[DEBUG]2020-06-20 05:29:29,939:utils:grad info pg1: norm std(0.000816) mean(0.001483)
[DEBUG]2020-06-20 05:29:30,236:utils:on_backward_begin lr: 3.1622776601683795e-05
[DEBUG]2020-06-20 05:29:30,238:utils:itr: 100, num_batch: 100, last loss: 0.661719, smooth_loss: 0.980239
[DEBUG]2020-06-20 05:29:30,246:utils:loss_avg: 1.03827, lr_pg0:3.1622776601683795e-05, lr_pg1: 3.1622776601683795e-05final_score:0.51834, mc_score:0.04570
[DEBUG]2020-06-20 05:29:30,613:utils:grad info pg0: norm std(0.003917) mean(0.011920)
[DEBUG]2020-06-20 05:29:30,616:utils:grad info pg1: norm std(0.000579) mean(0.001057)
[DEBUG]2020-06-20 05:29:30,911:utils:on_backward_begin lr: 3.349654391578277e-05
[DEBUG]2020-06-20 05:29:30,913:utils:itr: 101, num_batch: 101, last loss: 0.697016, smooth_loss: 0.973748
[DEBUG]2020-06-20 05:29:30,922:utils:loss_avg: 1.03492, lr_pg0:3.349654391578277e-05, lr_pg1: 3.349654391578277e-05final_score:0.52256, mc_score:0.05044
[DEBUG]2020-06-20 05:29:31,289:utils:grad info pg0: norm std(0.004674) mean(0.012856)
[DEBUG]2020-06-20 05:29:31,293:utils:grad info pg1: norm std(0.000714) mean(0.001284)
[DEBUG]2020-06-20 05:29:31,604:utils:on_backward_begin lr: 3.548133892335755e-05
[DEBUG]2020-06-20 05:29:31,605:utils:itr: 102, num_batch: 102, last loss: 0.609142, smooth_loss: 0.965415
[DEBUG]2020-06-20 05:29:31,615:utils:loss_avg: 1.03079, lr_pg0:3.548133892335755e-05, lr_pg1: 3.548133892335755e-05final_score:0.52919, mc_score:0.06166
[DEBUG]2020-06-20 05:29:31,983:utils:grad info pg0: norm std(0.002645) mean(0.007443)
[DEBUG]2020-06-20 05:29:31,987:utils:grad info pg1: norm std(0.000470) mean(0.000807)
[DEBUG]2020-06-20 05:29:32,284:utils:on_backward_begin lr: 3.7583740428844426e-05
[DEBUG]2020-06-20 05:29:32,286:utils:itr: 103, num_batch: 103, last loss: 0.577015, smooth_loss: 0.956565
[DEBUG]2020-06-20 05:29:32,298:utils:loss_avg: 1.02642, lr_pg0:3.7583740428844426e-05, lr_pg1: 3.7583740428844426e-05final_score:0.53554, mc_score:0.06985
[DEBUG]2020-06-20 05:29:32,667:utils:grad info pg0: norm std(0.007346) mean(0.016093)
[DEBUG]2020-06-20 05:29:32,670:utils:grad info pg1: norm std(0.001049) mean(0.001869)
[DEBUG]2020-06-20 05:29:32,968:utils:on_backward_begin lr: 3.981071705534973e-05
[DEBUG]2020-06-20 05:29:32,969:utils:itr: 104, num_batch: 104, last loss: 1.401299, smooth_loss: 0.966671
[DEBUG]2020-06-20 05:29:32,979:utils:loss_avg: 1.02999, lr_pg0:3.981071705534973e-05, lr_pg1: 3.981071705534973e-05final_score:0.53309, mc_score:0.06543
[DEBUG]2020-06-20 05:29:33,348:utils:grad info pg0: norm std(0.004982) mean(0.008689)
[DEBUG]2020-06-20 05:29:33,351:utils:grad info pg1: norm std(0.000700) mean(0.001120)
[DEBUG]2020-06-20 05:29:33,664:utils:on_backward_begin lr: 4.216965034285823e-05
[DEBUG]2020-06-20 05:29:33,666:utils:itr: 105, num_batch: 105, last loss: 0.871520, smooth_loss: 0.964515
[DEBUG]2020-06-20 05:29:33,676:utils:loss_avg: 1.02850, lr_pg0:4.216965034285823e-05, lr_pg1: 4.216965034285823e-05final_score:0.53642, mc_score:0.07018
[DEBUG]2020-06-20 05:29:34,043:utils:grad info pg0: norm std(0.006455) mean(0.017108)
[DEBUG]2020-06-20 05:29:34,046:utils:grad info pg1: norm std(0.000878) mean(0.001621)
[DEBUG]2020-06-20 05:29:34,342:utils:on_backward_begin lr: 4.466835921509632e-05
[DEBUG]2020-06-20 05:29:34,343:utils:itr: 106, num_batch: 106, last loss: 0.791408, smooth_loss: 0.960602
[DEBUG]2020-06-20 05:29:34,353:utils:loss_avg: 1.02628, lr_pg0:4.466835921509632e-05, lr_pg1: 4.466835921509632e-05final_score:0.53729, mc_score:0.07099
[DEBUG]2020-06-20 05:29:34,721:utils:grad info pg0: norm std(0.004634) mean(0.014234)
[DEBUG]2020-06-20 05:29:34,725:utils:grad info pg1: norm std(0.000742) mean(0.001379)
[DEBUG]2020-06-20 05:29:35,021:utils:on_backward_begin lr: 4.731512589614806e-05
[DEBUG]2020-06-20 05:29:35,023:utils:itr: 107, num_batch: 107, last loss: 1.143250, smooth_loss: 0.964720
[DEBUG]2020-06-20 05:29:35,033:utils:loss_avg: 1.02737, lr_pg0:4.731512589614806e-05, lr_pg1: 4.731512589614806e-05final_score:0.53574, mc_score:0.06708
[DEBUG]2020-06-20 05:29:35,401:utils:grad info pg0: norm std(0.003619) mean(0.009609)
[DEBUG]2020-06-20 05:29:35,404:utils:grad info pg1: norm std(0.000618) mean(0.001063)
[DEBUG]2020-06-20 05:29:35,708:utils:on_backward_begin lr: 5.0118723362727245e-05
[DEBUG]2020-06-20 05:29:35,710:utils:itr: 108, num_batch: 108, last loss: 1.511854, smooth_loss: 0.977023
[DEBUG]2020-06-20 05:29:35,720:utils:loss_avg: 1.03181, lr_pg0:5.0118723362727245e-05, lr_pg1: 5.0118723362727245e-05final_score:0.53265, mc_score:0.06592
[DEBUG]2020-06-20 05:29:36,088:utils:grad info pg0: norm std(0.002860) mean(0.007668)
[DEBUG]2020-06-20 05:29:36,091:utils:grad info pg1: norm std(0.000356) mean(0.000641)
[DEBUG]2020-06-20 05:29:36,390:utils:on_backward_begin lr: 5.308844442309886e-05
[DEBUG]2020-06-20 05:29:36,392:utils:itr: 109, num_batch: 109, last loss: 1.033669, smooth_loss: 0.978293
[DEBUG]2020-06-20 05:29:36,403:utils:loss_avg: 1.03183, lr_pg0:5.308844442309886e-05, lr_pg1: 5.308844442309886e-05final_score:0.53241, mc_score:0.06232
[DEBUG]2020-06-20 05:29:36,772:utils:grad info pg0: norm std(0.002206) mean(0.005599)
[DEBUG]2020-06-20 05:29:36,775:utils:grad info pg1: norm std(0.000376) mean(0.000638)
[DEBUG]2020-06-20 05:29:37,070:utils:on_backward_begin lr: 5.6234132519034934e-05
[DEBUG]2020-06-20 05:29:37,072:utils:itr: 110, num_batch: 110, last loss: 0.753197, smooth_loss: 0.973257
[DEBUG]2020-06-20 05:29:37,081:utils:loss_avg: 1.02932, lr_pg0:5.6234132519034934e-05, lr_pg1: 5.6234132519034934e-05final_score:0.53455, mc_score:0.06629
[DEBUG]2020-06-20 05:29:37,449:utils:grad info pg0: norm std(0.003564) mean(0.008525)
[DEBUG]2020-06-20 05:29:37,452:utils:grad info pg1: norm std(0.000426) mean(0.000826)
[DEBUG]2020-06-20 05:29:37,751:utils:on_backward_begin lr: 5.9566214352901076e-05
[DEBUG]2020-06-20 05:29:37,753:utils:itr: 111, num_batch: 111, last loss: 1.171997, smooth_loss: 0.977693
[DEBUG]2020-06-20 05:29:37,762:utils:loss_avg: 1.03059, lr_pg0:5.9566214352901076e-05, lr_pg1: 5.9566214352901076e-05final_score:0.53309, mc_score:0.06277
[DEBUG]2020-06-20 05:29:38,136:utils:grad info pg0: norm std(0.014601) mean(0.029192)
[DEBUG]2020-06-20 05:29:38,140:utils:grad info pg1: norm std(0.001604) mean(0.002689)
[DEBUG]2020-06-20 05:29:38,451:utils:on_backward_begin lr: 6.309573444801936e-05
[DEBUG]2020-06-20 05:29:38,452:utils:itr: 112, num_batch: 112, last loss: 1.360550, smooth_loss: 0.986220
[DEBUG]2020-06-20 05:29:38,463:utils:loss_avg: 1.03351, lr_pg0:6.309573444801936e-05, lr_pg1: 6.309573444801936e-05final_score:0.52924, mc_score:0.05599
[DEBUG]2020-06-20 05:29:38,831:utils:grad info pg0: norm std(0.002697) mean(0.007250)
[DEBUG]2020-06-20 05:29:38,835:utils:grad info pg1: norm std(0.000391) mean(0.000702)

[DEBUG]2020-06-20 05:29:39,135:utils:on_backward_begin lr: 6.683439175686141e-05
[DEBUG]2020-06-20 05:29:39,136:utils:itr: 113, num_batch: 113, last loss: 0.887789, smooth_loss: 0.984033
[DEBUG]2020-06-20 05:29:39,147:utils:loss_avg: 1.03223, lr_pg0:6.683439175686141e-05, lr_pg1: 6.683439175686141e-05final_score:0.52988, mc_score:0.05538
[DEBUG]2020-06-20 05:29:39,516:utils:grad info pg0: norm std(0.007396) mean(0.016888)
[DEBUG]2020-06-20 05:29:39,519:utils:grad info pg1: norm std(0.000830) mean(0.001519)
[DEBUG]2020-06-20 05:29:39,818:utils:on_backward_begin lr: 7.079457843841374e-05
[DEBUG]2020-06-20 05:29:39,819:utils:itr: 114, num_batch: 114, last loss: 0.980857, smooth_loss: 0.983962
[DEBUG]2020-06-20 05:29:39,829:utils:loss_avg: 1.03179, lr_pg0:7.079457843841374e-05, lr_pg1: 7.079457843841374e-05final_score:0.52897, mc_score:0.05330
[DEBUG]2020-06-20 05:29:40,196:utils:grad info pg0: norm std(0.004894) mean(0.013910)
[DEBUG]2020-06-20 05:29:40,199:utils:grad info pg1: norm std(0.000781) mean(0.001428)
[DEBUG]2020-06-20 05:29:40,516:utils:on_backward_begin lr: 7.498942093324554e-05
[DEBUG]2020-06-20 05:29:40,518:utils:itr: 115, num_batch: 115, last loss: 0.891704, smooth_loss: 0.981921
[DEBUG]2020-06-20 05:29:40,528:utils:loss_avg: 1.03058, lr_pg0:7.498942093324554e-05, lr_pg1: 7.498942093324554e-05final_score:0.52887, mc_score:0.05093
[DEBUG]2020-06-20 05:29:40,902:utils:grad info pg0: norm std(0.005690) mean(0.012799)
[DEBUG]2020-06-20 05:29:40,906:utils:grad info pg1: norm std(0.000846) mean(0.001456)
[DEBUG]2020-06-20 05:29:41,226:utils:on_backward_begin lr: 7.943282347242811e-05
[DEBUG]2020-06-20 05:29:41,227:utils:itr: 116, num_batch: 116, last loss: 0.872181, smooth_loss: 0.979499
[DEBUG]2020-06-20 05:29:41,238:utils:loss_avg: 1.02922, lr_pg0:7.943282347242811e-05, lr_pg1: 7.943282347242811e-05final_score:0.52755, mc_score:0.04714
[DEBUG]2020-06-20 05:29:41,613:utils:grad info pg0: norm std(0.011947) mean(0.021821)
[DEBUG]2020-06-20 05:29:41,616:utils:grad info pg1: norm std(0.000961) mean(0.001882)
[DEBUG]2020-06-20 05:29:41,914:utils:on_backward_begin lr: 8.413951416451946e-05
[DEBUG]2020-06-20 05:29:41,916:utils:itr: 117, num_batch: 117, last loss: 0.779552, smooth_loss: 0.975094
[DEBUG]2020-06-20 05:29:41,926:utils:loss_avg: 1.02711, lr_pg0:8.413951416451946e-05, lr_pg1: 8.413951416451946e-05final_score:0.52927, mc_score:0.04869
[DEBUG]2020-06-20 05:29:42,293:utils:grad info pg0: norm std(0.006827) mean(0.017747)
[DEBUG]2020-06-20 05:29:42,296:utils:grad info pg1: norm std(0.000842) mean(0.001586)
[DEBUG]2020-06-20 05:29:42,592:utils:on_backward_begin lr: 8.912509381337452e-05
[DEBUG]2020-06-20 05:29:42,593:utils:itr: 118, num_batch: 118, last loss: 0.967416, smooth_loss: 0.974925
[DEBUG]2020-06-20 05:29:42,603:utils:loss_avg: 1.02661, lr_pg0:8.912509381337452e-05, lr_pg1: 8.912509381337452e-05final_score:0.52857, mc_score:0.04818
[DEBUG]2020-06-20 05:29:42,968:utils:grad info pg0: norm std(0.004974) mean(0.013891)
[DEBUG]2020-06-20 05:29:42,971:utils:grad info pg1: norm std(0.000648) mean(0.001244)
[DEBUG]2020-06-20 05:29:43,275:utils:on_backward_begin lr: 9.44060876285923e-05
[DEBUG]2020-06-20 05:29:43,276:utils:itr: 119, num_batch: 119, last loss: 0.710971, smooth_loss: 0.969133
[DEBUG]2020-06-20 05:29:43,286:utils:loss_avg: 1.02398, lr_pg0:9.44060876285923e-05, lr_pg1: 9.44060876285923e-05final_score:0.53144, mc_score:0.05280
[DEBUG]2020-06-20 05:29:43,654:utils:grad info pg0: norm std(0.005180) mean(0.011670)
[DEBUG]2020-06-20 05:29:43,657:utils:grad info pg1: norm std(0.000609) mean(0.001131)
[DEBUG]2020-06-20 05:29:43,958:utils:on_backward_begin lr: 9.999999999999998e-05
[DEBUG]2020-06-20 05:29:43,960:utils:itr: 120, num_batch: 120, last loss: 0.706354, smooth_loss: 0.963378
[DEBUG]2020-06-20 05:29:43,969:utils:loss_avg: 1.02135, lr_pg0:9.999999999999998e-05, lr_pg1: 9.999999999999998e-05final_score:0.53393, mc_score:0.05651
[DEBUG]2020-06-20 05:29:44,335:utils:grad info pg0: norm std(0.004700) mean(0.011833)
[DEBUG]2020-06-20 05:29:44,338:utils:grad info pg1: norm std(0.000943) mean(0.001489)
[DEBUG]2020-06-20 05:29:44,650:utils:on_backward_begin lr: 0.00010592537251772888
[DEBUG]2020-06-20 05:29:44,651:utils:itr: 121, num_batch: 121, last loss: 0.981949, smooth_loss: 0.963784
[DEBUG]2020-06-20 05:29:44,660:utils:loss_avg: 1.02103, lr_pg0:0.00010592537251772888, lr_pg1: 0.00010592537251772888final_score:0.53278, mc_score:0.05269
[DEBUG]2020-06-20 05:29:45,029:utils:grad info pg0: norm std(0.004012) mean(0.010665)
[DEBUG]2020-06-20 05:29:45,032:utils:grad info pg1: norm std(0.000841) mean(0.001394)
[DEBUG]2020-06-20 05:29:45,342:utils:on_backward_begin lr: 0.00011220184543019631
[DEBUG]2020-06-20 05:29:45,344:utils:itr: 122, num_batch: 122, last loss: 0.838404, smooth_loss: 0.961048
[DEBUG]2020-06-20 05:29:45,356:utils:loss_avg: 1.01954, lr_pg0:0.00011220184543019631, lr_pg1: 0.00011220184543019631final_score:0.53248, mc_score:0.04944
[DEBUG]2020-06-20 05:29:45,772:utils:grad info pg0: norm std(0.035255) mean(0.036511)
[DEBUG]2020-06-20 05:29:45,775:utils:grad info pg1: norm std(0.001304) mean(0.002387)
[DEBUG]2020-06-20 05:29:46,075:utils:on_backward_begin lr: 0.00011885022274370182
[DEBUG]2020-06-20 05:29:46,076:utils:itr: 123, num_batch: 123, last loss: 0.622357, smooth_loss: 0.953672
[DEBUG]2020-06-20 05:29:46,087:utils:loss_avg: 1.01634, lr_pg0:0.00011885022274370182, lr_pg1: 0.00011885022274370182final_score:0.53640, mc_score:0.05251
[DEBUG]2020-06-20 05:29:46,455:utils:grad info pg0: norm std(0.013064) mean(0.034433)
[DEBUG]2020-06-20 05:29:46,458:utils:grad info pg1: norm std(0.001775) mean(0.003155)
[DEBUG]2020-06-20 05:29:46,754:utils:on_backward_begin lr: 0.0001258925411794167
[DEBUG]2020-06-20 05:29:46,756:utils:itr: 124, num_batch: 124, last loss: 1.030403, smooth_loss: 0.955340
[DEBUG]2020-06-20 05:29:46,765:utils:loss_avg: 1.01645, lr_pg0:0.0001258925411794167, lr_pg1: 0.0001258925411794167final_score:0.53702, mc_score:0.05191
[DEBUG]2020-06-20 05:29:47,133:utils:grad info pg0: norm std(0.008783) mean(0.019647)
[DEBUG]2020-06-20 05:29:47,137:utils:grad info pg1: norm std(0.000972) mean(0.001724)
[DEBUG]2020-06-20 05:29:47,440:utils:on_backward_begin lr: 0.0001333521432163324
[DEBUG]2020-06-20 05:29:47,442:utils:itr: 125, num_batch: 125, last loss: 0.818678, smooth_loss: 0.952375
[DEBUG]2020-06-20 05:29:47,453:utils:loss_avg: 1.01488, lr_pg0:0.0001333521432163324, lr_pg1: 0.0001333521432163324final_score:0.53956, mc_score:0.05211
[DEBUG]2020-06-20 05:29:47,823:utils:grad info pg0: norm std(0.007203) mean(0.016430)
[DEBUG]2020-06-20 05:29:47,827:utils:grad info pg1: norm std(0.001546) mean(0.002389)
[DEBUG]2020-06-20 05:29:48,128:utils:on_backward_begin lr: 0.00014125375446227543
[DEBUG]2020-06-20 05:29:48,130:utils:itr: 126, num_batch: 126, last loss: 1.136188, smooth_loss: 0.956357
[DEBUG]2020-06-20 05:29:48,140:utils:loss_avg: 1.01584, lr_pg0:0.00014125375446227543, lr_pg1: 0.00014125375446227543final_score:0.54220, mc_score:0.05544
[DEBUG]2020-06-20 05:29:48,514:utils:grad info pg0: norm std(0.008423) mean(0.020125)
[DEBUG]2020-06-20 05:29:48,518:utils:grad info pg1: norm std(0.001457) mean(0.002326)
[DEBUG]2020-06-20 05:29:48,815:utils:on_backward_begin lr: 0.00014962356560944336
[DEBUG]2020-06-20 05:29:48,816:utils:itr: 127, num_batch: 127, last loss: 0.899351, smooth_loss: 0.955124
[DEBUG]2020-06-20 05:29:48,827:utils:loss_avg: 1.01493, lr_pg0:0.00014962356560944336, lr_pg1: 0.00014962356560944336final_score:0.54269, mc_score:0.05456
[DEBUG]2020-06-20 05:29:49,199:utils:grad info pg0: norm std(0.011918) mean(0.034453)
[DEBUG]2020-06-20 05:29:49,202:utils:grad info pg1: norm std(0.001358) mean(0.002691)
[DEBUG]2020-06-20 05:29:49,507:utils:on_backward_begin lr: 0.00015848931924611136
[DEBUG]2020-06-20 05:29:49,509:utils:itr: 128, num_batch: 128, last loss: 0.997433, smooth_loss: 0.956038
[DEBUG]2020-06-20 05:29:49,518:utils:loss_avg: 1.01479, lr_pg0:0.00015848931924611136, lr_pg1: 0.00015848931924611136final_score:0.54273, mc_score:0.05265
[DEBUG]2020-06-20 05:29:49,886:utils:grad info pg0: norm std(0.006196) mean(0.015010)
[DEBUG]2020-06-20 05:29:49,889:utils:grad info pg1: norm std(0.000945) mean(0.001704)

[DEBUG]2020-06-20 05:29:50,191:utils:on_backward_begin lr: 0.00016788040181225605
[DEBUG]2020-06-20 05:29:50,192:utils:itr: 129, num_batch: 129, last loss: 1.000897, smooth_loss: 0.957005
[DEBUG]2020-06-20 05:29:50,206:utils:loss_avg: 1.01469, lr_pg0:0.00016788040181225605, lr_pg1: 0.00016788040181225605final_score:0.54333, mc_score:0.05449
[DEBUG]2020-06-20 05:29:50,573:utils:grad info pg0: norm std(0.008089) mean(0.020454)
[DEBUG]2020-06-20 05:29:50,577:utils:grad info pg1: norm std(0.001405) mean(0.002338)
[DEBUG]2020-06-20 05:29:50,881:utils:on_backward_begin lr: 0.00017782794100389232
[DEBUG]2020-06-20 05:29:50,883:utils:itr: 130, num_batch: 130, last loss: 0.693575, smooth_loss: 0.951334
[DEBUG]2020-06-20 05:29:50,894:utils:loss_avg: 1.01224, lr_pg0:0.00017782794100389232, lr_pg1: 0.00017782794100389232final_score:0.54667, mc_score:0.05782
[DEBUG]2020-06-20 05:29:51,262:utils:grad info pg0: norm std(0.009449) mean(0.019268)
[DEBUG]2020-06-20 05:29:51,265:utils:grad info pg1: norm std(0.001201) mean(0.002101)
[DEBUG]2020-06-20 05:29:51,576:utils:on_backward_begin lr: 0.0001883649089489801
[DEBUG]2020-06-20 05:29:51,578:utils:itr: 131, num_batch: 131, last loss: 0.701814, smooth_loss: 0.945971
[DEBUG]2020-06-20 05:29:51,588:utils:loss_avg: 1.00988, lr_pg0:0.0001883649089489801, lr_pg1: 0.0001883649089489801final_score:0.54932, mc_score:0.06163
[DEBUG]2020-06-20 05:29:51,958:utils:grad info pg0: norm std(0.000865) mean(0.002146)
[DEBUG]2020-06-20 05:29:51,962:utils:grad info pg1: norm std(0.000150) mean(0.000252)
[DEBUG]2020-06-20 05:29:52,260:utils:on_backward_begin lr: 0.000199526231496888
[DEBUG]2020-06-20 05:29:52,263:utils:itr: 132, num_batch: 132, last loss: 0.677522, smooth_loss: 0.940210
[DEBUG]2020-06-20 05:29:52,273:utils:loss_avg: 1.00738, lr_pg0:0.000199526231496888, lr_pg1: 0.000199526231496888final_score:0.55250, mc_score:0.06568
[DEBUG]2020-06-20 05:29:52,641:utils:grad info pg0: norm std(0.000117) mean(0.000200)
[DEBUG]2020-06-20 05:29:52,644:utils:grad info pg1: norm std(0.000011) mean(0.000020)
[DEBUG]2020-06-20 05:29:52,945:utils:on_backward_begin lr: 0.0002113489039836648
[DEBUG]2020-06-20 05:29:52,946:utils:itr: 133, num_batch: 133, last loss: 1.798826, smooth_loss: 0.958610
[DEBUG]2020-06-20 05:29:52,957:utils:loss_avg: 1.01329, lr_pg0:0.0002113489039836648, lr_pg1: 0.0002113489039836648final_score:0.54946, mc_score:0.06245
[DEBUG]2020-06-20 05:29:53,325:utils:grad info pg0: norm std(0.000187) mean(0.000387)
[DEBUG]2020-06-20 05:29:53,329:utils:grad info pg1: norm std(0.000019) mean(0.000034)
[DEBUG]2020-06-20 05:29:53,642:utils:on_backward_begin lr: 0.00022387211385683408
[DEBUG]2020-06-20 05:29:53,643:utils:itr: 134, num_batch: 134, last loss: 2.004953, smooth_loss: 0.981001
[DEBUG]2020-06-20 05:29:53,654:utils:loss_avg: 1.02064, lr_pg0:0.00022387211385683408, lr_pg1: 0.00022387211385683408final_score:0.54641, mc_score:0.05933
[DEBUG]2020-06-20 05:29:54,022:utils:grad info pg0: norm std(0.000051) mean(0.000111)
[DEBUG]2020-06-20 05:29:54,025:utils:grad info pg1: norm std(0.000006) mean(0.000011)
[DEBUG]2020-06-20 05:29:54,324:utils:on_backward_begin lr: 0.00023713737056616565
[DEBUG]2020-06-20 05:29:54,325:utils:itr: 135, num_batch: 135, last loss: 1.449392, smooth_loss: 0.991010
[DEBUG]2020-06-20 05:29:54,335:utils:loss_avg: 1.02379, lr_pg0:0.00023713737056616565, lr_pg1: 0.00023713737056616565final_score:0.54154, mc_score:0.05200
[DEBUG]2020-06-20 05:29:54,704:utils:grad info pg0: norm std(0.000014) mean(0.000024)
[DEBUG]2020-06-20 05:29:54,708:utils:grad info pg1: norm std(0.000001) mean(0.000003)
[DEBUG]2020-06-20 05:29:55,006:utils:on_backward_begin lr: 0.00025118864315095817
[DEBUG]2020-06-20 05:29:55,008:utils:itr: 136, num_batch: 136, last loss: 0.813053, smooth_loss: 0.987213
[DEBUG]2020-06-20 05:29:55,018:utils:loss_avg: 1.02225, lr_pg0:0.00025118864315095817, lr_pg1: 0.00025118864315095817final_score:0.54199, mc_score:0.05162
[DEBUG]2020-06-20 05:29:55,387:utils:grad info pg0: norm std(0.000064) mean(0.000137)
[DEBUG]2020-06-20 05:29:55,390:utils:grad info pg1: norm std(0.000006) mean(0.000012)
[DEBUG]2020-06-20 05:29:55,688:utils:on_backward_begin lr: 0.0002660725059798811
[DEBUG]2020-06-20 05:29:55,689:utils:itr: 137, num_batch: 137, last loss: 0.833798, smooth_loss: 0.983943
[DEBUG]2020-06-20 05:29:55,700:utils:loss_avg: 1.02089, lr_pg0:0.0002660725059798811, lr_pg1: 0.0002660725059798811final_score:0.54169, mc_score:0.05084
[DEBUG]2020-06-20 05:29:56,073:utils:grad info pg0: norm std(0.000049) mean(0.000090)
[DEBUG]2020-06-20 05:29:56,076:utils:grad info pg1: norm std(0.000005) mean(0.000008)
[DEBUG]2020-06-20 05:29:56,376:utils:on_backward_begin lr: 0.0002818382931264452
[DEBUG]2020-06-20 05:29:56,378:utils:itr: 138, num_batch: 138, last loss: 0.867028, smooth_loss: 0.981455
[DEBUG]2020-06-20 05:29:56,390:utils:loss_avg: 1.01978, lr_pg0:0.0002818382931264452, lr_pg1: 0.0002818382931264452final_score:0.54366, mc_score:0.05382
[DEBUG]2020-06-20 05:29:56,759:utils:grad info pg0: norm std(0.000052) mean(0.000106)
[DEBUG]2020-06-20 05:29:56,763:utils:grad info pg1: norm std(0.000005) mean(0.000009)
[DEBUG]2020-06-20 05:29:57,062:utils:on_backward_begin lr: 0.0002985382618917958
[DEBUG]2020-06-20 05:29:57,064:utils:itr: 139, num_batch: 139, last loss: 1.126376, smooth_loss: 0.984535
[DEBUG]2020-06-20 05:29:57,073:utils:loss_avg: 1.02054, lr_pg0:0.0002985382618917958, lr_pg1: 0.0002985382618917958final_score:0.54325, mc_score:0.05520
[DEBUG]2020-06-20 05:29:57,441:utils:grad info pg0: norm std(0.000065) mean(0.000152)
[DEBUG]2020-06-20 05:29:57,444:utils:grad info pg1: norm std(0.000008) mean(0.000015)
[DEBUG]2020-06-20 05:29:57,744:utils:on_backward_begin lr: 0.0003162277660168378
[DEBUG]2020-06-20 05:29:57,746:utils:itr: 140, num_batch: 140, last loss: 1.640198, smooth_loss: 0.998455
[DEBUG]2020-06-20 05:29:57,755:utils:loss_avg: 1.02493, lr_pg0:0.0003162277660168378, lr_pg1: 0.0003162277660168378final_score:0.53922, mc_score:0.05053
[DEBUG]2020-06-20 05:29:58,123:utils:grad info pg0: norm std(0.000047) mean(0.000081)
[DEBUG]2020-06-20 05:29:58,126:utils:grad info pg1: norm std(0.000005) mean(0.000009)
[DEBUG]2020-06-20 05:29:58,425:utils:on_backward_begin lr: 0.00033496543915782746
[DEBUG]2020-06-20 05:29:58,427:utils:itr: 141, num_batch: 141, last loss: 1.014719, smooth_loss: 0.998800
[DEBUG]2020-06-20 05:29:58,436:utils:loss_avg: 1.02486, lr_pg0:0.00033496543915782746, lr_pg1: 0.00033496543915782746final_score:0.53794, mc_score:0.04878
[DEBUG]2020-06-20 05:29:58,804:utils:grad info pg0: norm std(0.000133) mean(0.000223)
[DEBUG]2020-06-20 05:29:58,808:utils:grad info pg1: norm std(0.000009) mean(0.000018)
[DEBUG]2020-06-20 05:29:59,105:utils:on_backward_begin lr: 0.00035481338923357527
[DEBUG]2020-06-20 05:29:59,106:utils:itr: 142, num_batch: 142, last loss: 0.723214, smooth_loss: 0.992963
[DEBUG]2020-06-20 05:29:59,116:utils:loss_avg: 1.02275, lr_pg0:0.00035481338923357527, lr_pg1: 0.00035481338923357527final_score:0.53884, mc_score:0.05172
[DEBUG]2020-06-20 05:29:59,485:utils:grad info pg0: norm std(0.000253) mean(0.000369)
[DEBUG]2020-06-20 05:29:59,487:utils:grad info pg1: norm std(0.000027) mean(0.000046)
[DEBUG]2020-06-20 05:29:59,786:utils:on_backward_begin lr: 0.000375837404288444
[DEBUG]2020-06-20 05:29:59,788:utils:itr: 143, num_batch: 143, last loss: 0.849595, smooth_loss: 0.989931
[DEBUG]2020-06-20 05:29:59,797:utils:loss_avg: 1.02155, lr_pg0:0.000375837404288444, lr_pg1: 0.000375837404288444final_score:0.53881, mc_score:0.05116
[DEBUG]2020-06-20 05:30:00,164:utils:grad info pg0: norm std(0.000239) mean(0.000418)
[DEBUG]2020-06-20 05:30:00,168:utils:grad info pg1: norm std(0.000023) mean(0.000041)
[DEBUG]2020-06-20 05:30:00,468:utils:on_backward_begin lr: 0.00039810717055349714
[DEBUG]2020-06-20 05:30:00,469:utils:itr: 144, num_batch: 144, last loss: 0.937151, smooth_loss: 0.988816
[DEBUG]2020-06-20 05:30:00,478:utils:loss_avg: 1.02097, lr_pg0:0.00039810717055349714, lr_pg1: 0.00039810717055349714final_score:0.53743, mc_score:0.04692
[DEBUG]2020-06-20 05:30:00,847:utils:grad info pg0: norm std(0.000243) mean(0.000427)
[DEBUG]2020-06-20 05:30:00,851:utils:grad info pg1: norm std(0.000025) mean(0.000045)

[DEBUG]2020-06-20 05:30:01,152:utils:on_backward_begin lr: 0.00042169650342858213
[DEBUG]2020-06-20 05:30:01,153:utils:itr: 145, num_batch: 145, last loss: 0.782527, smooth_loss: 0.984462
[DEBUG]2020-06-20 05:30:01,163:utils:loss_avg: 1.01934, lr_pg0:0.00042169650342858213, lr_pg1: 0.00042169650342858213final_score:0.53844, mc_score:0.04972
[DEBUG]2020-06-20 05:30:01,530:utils:grad info pg0: norm std(0.000114) mean(0.000184)
[DEBUG]2020-06-20 05:30:01,533:utils:grad info pg1: norm std(0.000012) mean(0.000021)
[DEBUG]2020-06-20 05:30:01,833:utils:on_backward_begin lr: 0.00044668359215096305
[DEBUG]2020-06-20 05:30:01,835:utils:itr: 146, num_batch: 146, last loss: 1.084105, smooth_loss: 0.986563
[DEBUG]2020-06-20 05:30:01,845:utils:loss_avg: 1.01978, lr_pg0:0.00044668359215096305, lr_pg1: 0.00044668359215096305final_score:0.53752, mc_score:0.04959
[DEBUG]2020-06-20 05:30:02,219:utils:grad info pg0: norm std(0.000333) mean(0.000472)
[DEBUG]2020-06-20 05:30:02,223:utils:grad info pg1: norm std(0.000023) mean(0.000042)
[DEBUG]2020-06-20 05:30:02,519:utils:on_backward_begin lr: 0.00047315125896148035
[DEBUG]2020-06-20 05:30:02,520:utils:itr: 147, num_batch: 147, last loss: 1.166298, smooth_loss: 0.990348
[DEBUG]2020-06-20 05:30:02,530:utils:loss_avg: 1.02077, lr_pg0:0.00047315125896148035, lr_pg1: 0.00047315125896148035final_score:0.53606, mc_score:0.04663
[DEBUG]2020-06-20 05:30:02,895:utils:grad info pg0: norm std(0.000043) mean(0.000072)
[DEBUG]2020-06-20 05:30:02,898:utils:grad info pg1: norm std(0.000004) mean(0.000007)
[DEBUG]2020-06-20 05:30:03,194:utils:on_backward_begin lr: 0.0005011872336272722
[DEBUG]2020-06-20 05:30:03,195:utils:itr: 148, num_batch: 148, last loss: 0.719493, smooth_loss: 0.984650
[DEBUG]2020-06-20 05:30:03,205:utils:loss_avg: 1.01874, lr_pg0:0.0005011872336272722, lr_pg1: 0.0005011872336272722final_score:0.53812, mc_score:0.05198
[DEBUG]2020-06-20 05:30:03,571:utils:grad info pg0: norm std(0.000029) mean(0.000048)
[DEBUG]2020-06-20 05:30:03,574:utils:grad info pg1: norm std(0.000003) mean(0.000006)
[DEBUG]2020-06-20 05:30:03,872:utils:on_backward_begin lr: 0.0005308844442309883
[DEBUG]2020-06-20 05:30:03,873:utils:itr: 149, num_batch: 149, last loss: 2.115553, smooth_loss: 1.008416
[DEBUG]2020-06-20 05:30:03,883:utils:loss_avg: 1.02606, lr_pg0:0.0005308844442309883, lr_pg1: 0.0005308844442309883final_score:0.53578, mc_score:0.04933
[DEBUG]2020-06-20 05:30:04,252:utils:grad info pg0: norm std(0.000016) mean(0.000028)
[DEBUG]2020-06-20 05:30:04,256:utils:grad info pg1: norm std(0.000002) mean(0.000003)
[DEBUG]2020-06-20 05:30:04,562:utils:on_backward_begin lr: 0.0005623413251903491
[DEBUG]2020-06-20 05:30:04,564:utils:itr: 150, num_batch: 150, last loss: 2.566210, smooth_loss: 1.041119
[DEBUG]2020-06-20 05:30:04,574:utils:loss_avg: 1.03626, lr_pg0:0.0005623413251903491, lr_pg1: 0.0005623413251903491final_score:0.53033, mc_score:0.04288
[DEBUG]2020-06-20 05:30:04,942:utils:grad info pg0: norm std(0.000003) mean(0.000005)
[DEBUG]2020-06-20 05:30:04,945:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:05,244:utils:on_backward_begin lr: 0.0005956621435290105
[DEBUG]2020-06-20 05:30:05,246:utils:itr: 151, num_batch: 151, last loss: 1.133912, smooth_loss: 1.043066
[DEBUG]2020-06-20 05:30:05,256:utils:loss_avg: 1.03690, lr_pg0:0.0005956621435290105, lr_pg1: 0.0005956621435290105final_score:0.53109, mc_score:0.04424
[DEBUG]2020-06-20 05:30:05,636:utils:grad info pg0: norm std(0.000001) mean(0.000002)
[DEBUG]2020-06-20 05:30:05,640:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:05,940:utils:on_backward_begin lr: 0.0006309573444801933
[DEBUG]2020-06-20 05:30:05,941:utils:itr: 152, num_batch: 152, last loss: 1.033337, smooth_loss: 1.042862
[DEBUG]2020-06-20 05:30:05,952:utils:loss_avg: 1.03688, lr_pg0:0.0006309573444801933, lr_pg1: 0.0006309573444801933final_score:0.52920, mc_score:0.03833
[DEBUG]2020-06-20 05:30:06,320:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:06,324:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:06,626:utils:on_backward_begin lr: 0.0006683439175686147
[DEBUG]2020-06-20 05:30:06,627:utils:itr: 153, num_batch: 153, last loss: 0.974306, smooth_loss: 1.041427
[DEBUG]2020-06-20 05:30:06,637:utils:loss_avg: 1.03647, lr_pg0:0.0006683439175686147, lr_pg1: 0.0006683439175686147final_score:0.52842, mc_score:0.03686
[DEBUG]2020-06-20 05:30:07,005:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:07,009:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:07,308:utils:on_backward_begin lr: 0.000707945784384138
[DEBUG]2020-06-20 05:30:07,310:utils:itr: 154, num_batch: 154, last loss: 1.095729, smooth_loss: 1.042562
[DEBUG]2020-06-20 05:30:07,319:utils:loss_avg: 1.03685, lr_pg0:0.000707945784384138, lr_pg1: 0.000707945784384138final_score:0.52725, mc_score:0.03422
[DEBUG]2020-06-20 05:30:07,686:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:07,690:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:07,993:utils:on_backward_begin lr: 0.000749894209332456
[DEBUG]2020-06-20 05:30:07,994:utils:itr: 155, num_batch: 155, last loss: 0.947869, smooth_loss: 1.040584
[DEBUG]2020-06-20 05:30:08,004:utils:loss_avg: 1.03628, lr_pg0:0.000749894209332456, lr_pg1: 0.000749894209332456final_score:0.52690, mc_score:0.02977
[DEBUG]2020-06-20 05:30:08,372:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:08,375:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:08,677:utils:on_backward_begin lr: 0.0007943282347242816
[DEBUG]2020-06-20 05:30:08,678:utils:itr: 156, num_batch: 156, last loss: 0.945866, smooth_loss: 1.038607
[DEBUG]2020-06-20 05:30:08,688:utils:loss_avg: 1.03571, lr_pg0:0.0007943282347242816, lr_pg1: 0.0007943282347242816final_score:0.52605, mc_score:0.02744
[DEBUG]2020-06-20 05:30:09,056:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:09,059:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:09,357:utils:on_backward_begin lr: 0.0008413951416451952
[DEBUG]2020-06-20 05:30:09,359:utils:itr: 157, num_batch: 157, last loss: 0.621217, smooth_loss: 1.029901
[DEBUG]2020-06-20 05:30:09,370:utils:loss_avg: 1.03308, lr_pg0:0.0008413951416451952, lr_pg1: 0.0008413951416451952final_score:0.52879, mc_score:0.03246
[DEBUG]2020-06-20 05:30:09,744:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:09,747:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:10,058:utils:on_backward_begin lr: 0.0008912509381337458
[DEBUG]2020-06-20 05:30:10,060:utils:itr: 158, num_batch: 158, last loss: 1.010131, smooth_loss: 1.029489
[DEBUG]2020-06-20 05:30:10,073:utils:loss_avg: 1.03294, lr_pg0:0.0008912509381337458, lr_pg1: 0.0008912509381337458final_score:0.52951, mc_score:0.03375
[DEBUG]2020-06-20 05:30:10,446:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:10,450:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:10,762:utils:on_backward_begin lr: 0.0009440608762859237
[DEBUG]2020-06-20 05:30:10,764:utils:itr: 159, num_batch: 159, last loss: 1.718415, smooth_loss: 1.043834
[DEBUG]2020-06-20 05:30:10,774:utils:loss_avg: 1.03722, lr_pg0:0.0009440608762859237, lr_pg1: 0.0009440608762859237final_score:0.52625, mc_score:0.02968
[DEBUG]2020-06-20 05:30:11,149:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:11,152:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:11,461:utils:on_backward_begin lr: 0.0010000000000000005
[DEBUG]2020-06-20 05:30:11,463:utils:itr: 160, num_batch: 160, last loss: 1.335687, smooth_loss: 1.049906
[DEBUG]2020-06-20 05:30:11,473:utils:loss_avg: 1.03908, lr_pg0:0.0010000000000000005, lr_pg1: 0.0010000000000000005final_score:0.52440, mc_score:0.02747
[DEBUG]2020-06-20 05:30:11,841:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:11,844:utils:grad info pg1: norm std(0.000000) mean(0.000000)

[DEBUG]2020-06-20 05:30:12,145:utils:on_backward_begin lr: 0.0010592537251772895
[DEBUG]2020-06-20 05:30:12,146:utils:itr: 161, num_batch: 161, last loss: 1.083249, smooth_loss: 1.050599
[DEBUG]2020-06-20 05:30:12,157:utils:loss_avg: 1.03935, lr_pg0:0.0010592537251772895, lr_pg1: 0.0010592537251772895final_score:0.52326, mc_score:0.02532
[DEBUG]2020-06-20 05:30:12,530:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:12,534:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:12,845:utils:on_backward_begin lr: 0.0011220184543019641
[DEBUG]2020-06-20 05:30:12,847:utils:itr: 162, num_batch: 162, last loss: 0.935228, smooth_loss: 1.048203
[DEBUG]2020-06-20 05:30:12,858:utils:loss_avg: 1.03871, lr_pg0:0.0011220184543019641, lr_pg1: 0.0011220184543019641final_score:0.52291, mc_score:0.02400
[DEBUG]2020-06-20 05:30:13,225:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:13,228:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:13,524:utils:on_backward_begin lr: 0.0011885022274370177
[DEBUG]2020-06-20 05:30:13,525:utils:itr: 163, num_batch: 163, last loss: 0.824387, smooth_loss: 1.043557
[DEBUG]2020-06-20 05:30:13,534:utils:loss_avg: 1.03740, lr_pg0:0.0011885022274370177, lr_pg1: 0.0011885022274370177final_score:0.52333, mc_score:0.02541
[DEBUG]2020-06-20 05:30:13,902:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:13,905:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:14,203:utils:on_backward_begin lr: 0.0012589254117941666
[DEBUG]2020-06-20 05:30:14,205:utils:itr: 164, num_batch: 164, last loss: 0.922237, smooth_loss: 1.041041
[DEBUG]2020-06-20 05:30:14,216:utils:loss_avg: 1.03670, lr_pg0:0.0012589254117941666, lr_pg1: 0.0012589254117941666final_score:0.52340, mc_score:0.02411
[DEBUG]2020-06-20 05:30:14,584:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:14,587:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:14,901:utils:on_backward_begin lr: 0.0013335214321633232
[DEBUG]2020-06-20 05:30:14,903:utils:itr: 165, num_batch: 165, last loss: 0.891411, smooth_loss: 1.037940
[DEBUG]2020-06-20 05:30:14,912:utils:loss_avg: 1.03583, lr_pg0:0.0013335214321633232, lr_pg1: 0.0013335214321633232final_score:0.52294, mc_score:0.02259
[DEBUG]2020-06-20 05:30:15,282:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:15,285:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:15,781:utils:on_backward_begin lr: 0.0014125375446227537
[DEBUG]2020-06-20 05:30:15,782:utils:itr: 166, num_batch: 166, last loss: 0.810124, smooth_loss: 1.033222
[DEBUG]2020-06-20 05:30:15,792:utils:loss_avg: 1.03448, lr_pg0:0.0014125375446227537, lr_pg1: 0.0014125375446227537final_score:0.52325, mc_score:0.02389
[DEBUG]2020-06-20 05:30:16,160:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:16,163:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:16,484:utils:on_backward_begin lr: 0.0014962356560944327
[DEBUG]2020-06-20 05:30:16,486:utils:itr: 167, num_batch: 167, last loss: 0.962215, smooth_loss: 1.031753
[DEBUG]2020-06-20 05:30:16,496:utils:loss_avg: 1.03405, lr_pg0:0.0014962356560944327, lr_pg1: 0.0014962356560944327final_score:0.52372, mc_score:0.02516
[DEBUG]2020-06-20 05:30:16,865:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:16,868:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:17,171:utils:on_backward_begin lr: 0.001584893192461113
[DEBUG]2020-06-20 05:30:17,173:utils:itr: 168, num_batch: 168, last loss: 1.079736, smooth_loss: 1.032745
[DEBUG]2020-06-20 05:30:17,183:utils:loss_avg: 1.03432, lr_pg0:0.001584893192461113, lr_pg1: 0.001584893192461113final_score:0.52311, mc_score:0.02475
[DEBUG]2020-06-20 05:30:17,552:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:17,555:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:17,874:utils:on_backward_begin lr: 0.0016788040181225598
[DEBUG]2020-06-20 05:30:17,876:utils:itr: 169, num_batch: 169, last loss: 1.035752, smooth_loss: 1.032807
[DEBUG]2020-06-20 05:30:17,887:utils:loss_avg: 1.03433, lr_pg0:0.0016788040181225598, lr_pg1: 0.0016788040181225598final_score:0.52257, mc_score:0.02292
[DEBUG]2020-06-20 05:30:18,257:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:18,260:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:18,559:utils:on_backward_begin lr: 0.001778279410038922
[DEBUG]2020-06-20 05:30:18,561:utils:itr: 170, num_batch: 170, last loss: 0.851122, smooth_loss: 1.029055
[DEBUG]2020-06-20 05:30:18,571:utils:loss_avg: 1.03326, lr_pg0:0.001778279410038922, lr_pg1: 0.001778279410038922final_score:0.52271, mc_score:0.02431
[DEBUG]2020-06-20 05:30:18,939:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:18,942:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:19,238:utils:on_backward_begin lr: 0.0018836490894898
[DEBUG]2020-06-20 05:30:19,240:utils:itr: 171, num_batch: 171, last loss: 0.905630, smooth_loss: 1.026507
[DEBUG]2020-06-20 05:30:19,250:utils:loss_avg: 1.03251, lr_pg0:0.0018836490894898, lr_pg1: 0.0018836490894898final_score:0.52371, mc_score:0.02567
[DEBUG]2020-06-20 05:30:19,619:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:19,622:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:19,928:utils:on_backward_begin lr: 0.001995262314968879
[DEBUG]2020-06-20 05:30:19,930:utils:itr: 172, num_batch: 172, last loss: 1.524831, smooth_loss: 1.036786
[DEBUG]2020-06-20 05:30:19,940:utils:loss_avg: 1.03536, lr_pg0:0.001995262314968879, lr_pg1: 0.001995262314968879final_score:0.52135, mc_score:0.02313
[DEBUG]2020-06-20 05:30:20,308:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:20,311:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:20,612:utils:on_backward_begin lr: 0.002113489039836646
[DEBUG]2020-06-20 05:30:20,614:utils:itr: 173, num_batch: 173, last loss: 0.793858, smooth_loss: 1.031778
[DEBUG]2020-06-20 05:30:20,625:utils:loss_avg: 1.03397, lr_pg0:0.002113489039836646, lr_pg1: 0.002113489039836646final_score:0.52171, mc_score:0.02586
[DEBUG]2020-06-20 05:30:20,993:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:20,996:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:21,296:utils:on_backward_begin lr: 0.0022387211385683395
[DEBUG]2020-06-20 05:30:21,297:utils:itr: 174, num_batch: 174, last loss: 0.871456, smooth_loss: 1.028476
[DEBUG]2020-06-20 05:30:21,308:utils:loss_avg: 1.03304, lr_pg0:0.0022387211385683395, lr_pg1: 0.0022387211385683395final_score:0.52160, mc_score:0.02293
[DEBUG]2020-06-20 05:30:21,678:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:21,681:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:21,979:utils:on_backward_begin lr: 0.002371373705661655
[DEBUG]2020-06-20 05:30:21,980:utils:itr: 175, num_batch: 175, last loss: 1.095547, smooth_loss: 1.029857
[DEBUG]2020-06-20 05:30:21,991:utils:loss_avg: 1.03340, lr_pg0:0.002371373705661655, lr_pg1: 0.002371373705661655final_score:0.52046, mc_score:0.02096
[DEBUG]2020-06-20 05:30:22,359:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:22,362:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:22,659:utils:on_backward_begin lr: 0.00251188643150958
[DEBUG]2020-06-20 05:30:22,661:utils:itr: 176, num_batch: 176, last loss: 0.716887, smooth_loss: 1.023417
[DEBUG]2020-06-20 05:30:22,672:utils:loss_avg: 1.03161, lr_pg0:0.00251188643150958, lr_pg1: 0.00251188643150958final_score:0.52256, mc_score:0.02533
[DEBUG]2020-06-20 05:30:23,042:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:23,045:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:23,348:utils:on_backward_begin lr: 0.0026607250597988096

[DEBUG]2020-06-20 05:30:23,349:utils:itr: 177, num_batch: 177, last loss: 0.912919, smooth_loss: 1.021145
[DEBUG]2020-06-20 05:30:23,360:utils:loss_avg: 1.03094, lr_pg0:0.0026607250597988096, lr_pg1: 0.0026607250597988096final_score:0.52295, mc_score:0.02649
[DEBUG]2020-06-20 05:30:23,728:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:23,732:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:24,037:utils:on_backward_begin lr: 0.002818382931264454
[DEBUG]2020-06-20 05:30:24,038:utils:itr: 178, num_batch: 178, last loss: 1.114650, smooth_loss: 1.023067
[DEBUG]2020-06-20 05:30:24,050:utils:loss_avg: 1.03141, lr_pg0:0.002818382931264454, lr_pg1: 0.002818382931264454final_score:0.52064, mc_score:0.02148
[DEBUG]2020-06-20 05:30:24,419:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:24,423:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:24,720:utils:on_backward_begin lr: 0.0029853826189179603
[DEBUG]2020-06-20 05:30:24,721:utils:itr: 179, num_batch: 179, last loss: 1.316080, smooth_loss: 1.029085
[DEBUG]2020-06-20 05:30:24,731:utils:loss_avg: 1.03299, lr_pg0:0.0029853826189179603, lr_pg1: 0.0029853826189179603final_score:0.52066, mc_score:0.02155
[DEBUG]2020-06-20 05:30:25,098:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:25,102:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:25,402:utils:on_backward_begin lr: 0.0031622776601683803
[DEBUG]2020-06-20 05:30:25,404:utils:itr: 180, num_batch: 180, last loss: 0.587775, smooth_loss: 1.020025
[DEBUG]2020-06-20 05:30:25,414:utils:loss_avg: 1.03053, lr_pg0:0.0031622776601683803, lr_pg1: 0.0031622776601683803final_score:0.52371, mc_score:0.02535
[DEBUG]2020-06-20 05:30:25,783:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:25,787:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:26,085:utils:on_backward_begin lr: 0.0033496543915782777
[DEBUG]2020-06-20 05:30:26,086:utils:itr: 181, num_batch: 181, last loss: 1.243479, smooth_loss: 1.024610
[DEBUG]2020-06-20 05:30:26,095:utils:loss_avg: 1.03170, lr_pg0:0.0033496543915782777, lr_pg1: 0.0033496543915782777final_score:0.52340, mc_score:0.02537
[DEBUG]2020-06-20 05:30:26,463:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:26,466:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:26,769:utils:on_backward_begin lr: 0.003548133892335756
[DEBUG]2020-06-20 05:30:26,771:utils:itr: 182, num_batch: 182, last loss: 0.789365, smooth_loss: 1.019786
[DEBUG]2020-06-20 05:30:26,782:utils:loss_avg: 1.03038, lr_pg0:0.003548133892335756, lr_pg1: 0.003548133892335756final_score:0.52377, mc_score:0.02664
[DEBUG]2020-06-20 05:30:27,151:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:27,154:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:27,453:utils:on_backward_begin lr: 0.003758374042884443
[DEBUG]2020-06-20 05:30:27,455:utils:itr: 183, num_batch: 183, last loss: 0.955429, smooth_loss: 1.018467
[DEBUG]2020-06-20 05:30:27,466:utils:loss_avg: 1.02997, lr_pg0:0.003758374042884443, lr_pg1: 0.003758374042884443final_score:0.52374, mc_score:0.02629
[DEBUG]2020-06-20 05:30:27,834:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:27,837:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:28,139:utils:on_backward_begin lr: 0.003981071705534974
[DEBUG]2020-06-20 05:30:28,141:utils:itr: 184, num_batch: 184, last loss: 0.942730, smooth_loss: 1.016915
[DEBUG]2020-06-20 05:30:28,150:utils:loss_avg: 1.02950, lr_pg0:0.003981071705534974, lr_pg1: 0.003981071705534974final_score:0.52458, mc_score:0.02752
[DEBUG]2020-06-20 05:30:28,518:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:28,521:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:28,834:utils:on_backward_begin lr: 0.004216965034285824
[DEBUG]2020-06-20 05:30:28,835:utils:itr: 185, num_batch: 185, last loss: 0.774995, smooth_loss: 1.011961
[DEBUG]2020-06-20 05:30:28,845:utils:loss_avg: 1.02813, lr_pg0:0.004216965034285824, lr_pg1: 0.004216965034285824final_score:0.52535, mc_score:0.02885
[DEBUG]2020-06-20 05:30:29,211:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:29,215:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:29,516:utils:on_backward_begin lr: 0.004466835921509634
[DEBUG]2020-06-20 05:30:29,517:utils:itr: 186, num_batch: 186, last loss: 3.068631, smooth_loss: 1.054057
[DEBUG]2020-06-20 05:30:29,528:utils:loss_avg: 1.03904, lr_pg0:0.004466835921509634, lr_pg1: 0.004466835921509634final_score:0.52360, mc_score:0.02703
[DEBUG]2020-06-20 05:30:29,897:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:29,900:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:30,201:utils:on_backward_begin lr: 0.004731512589614808
[DEBUG]2020-06-20 05:30:30,203:utils:itr: 187, num_batch: 187, last loss: 1.263783, smooth_loss: 1.058348
[DEBUG]2020-06-20 05:30:30,213:utils:loss_avg: 1.04024, lr_pg0:0.004731512589614808, lr_pg1: 0.004731512589614808final_score:0.52335, mc_score:0.02670
[DEBUG]2020-06-20 05:30:30,581:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:30,584:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:30,891:utils:on_backward_begin lr: 0.005011872336272719
[DEBUG]2020-06-20 05:30:30,893:utils:itr: 188, num_batch: 188, last loss: 1.173010, smooth_loss: 1.060693
[DEBUG]2020-06-20 05:30:30,902:utils:loss_avg: 1.04094, lr_pg0:0.005011872336272719, lr_pg1: 0.005011872336272719final_score:0.52226, mc_score:0.02543
[DEBUG]2020-06-20 05:30:31,276:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:31,280:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:31,588:utils:on_backward_begin lr: 0.005308844442309881
[DEBUG]2020-06-20 05:30:31,590:utils:itr: 189, num_batch: 189, last loss: 0.922492, smooth_loss: 1.057868
[DEBUG]2020-06-20 05:30:31,601:utils:loss_avg: 1.04032, lr_pg0:0.005308844442309881, lr_pg1: 0.005308844442309881final_score:0.52220, mc_score:0.02400
[DEBUG]2020-06-20 05:30:31,970:utils:grad info pg0: norm std(0.000000) mean(0.000001)
[DEBUG]2020-06-20 05:30:31,973:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:32,271:utils:on_backward_begin lr: 0.0056234132519034875
[DEBUG]2020-06-20 05:30:32,273:utils:itr: 190, num_batch: 190, last loss: 3.406496, smooth_loss: 1.105853
[DEBUG]2020-06-20 05:30:32,284:utils:loss_avg: 1.05270, lr_pg0:0.0056234132519034875, lr_pg1: 0.0056234132519034875final_score:0.52051, mc_score:0.02225
[DEBUG]2020-06-20 05:30:32,655:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:32,658:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:32,959:utils:on_backward_begin lr: 0.005956621435290102
[DEBUG]2020-06-20 05:30:32,961:utils:itr: 191, num_batch: 191, last loss: 1.464909, smooth_loss: 1.113186
[DEBUG]2020-06-20 05:30:32,972:utils:loss_avg: 1.05485, lr_pg0:0.005956621435290102, lr_pg1: 0.005956621435290102final_score:0.51917, mc_score:0.02062
[DEBUG]2020-06-20 05:30:33,340:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:33,344:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:33,650:utils:on_backward_begin lr: 0.006309573444801929
[DEBUG]2020-06-20 05:30:33,653:utils:itr: 192, num_batch: 192, last loss: 2.378167, smooth_loss: 1.139008
[DEBUG]2020-06-20 05:30:33,663:utils:loss_avg: 1.06171, lr_pg0:0.006309573444801929, lr_pg1: 0.006309573444801929final_score:0.52040, mc_score:0.02183
[DEBUG]2020-06-20 05:30:34,033:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:34,036:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:34,337:utils:on_backward_begin lr: 0.006683439175686144
[DEBUG]2020-06-20 05:30:34,339:utils:itr: 193, num_batch: 193, last loss: 1.783793, smooth_loss: 1.152165

[DEBUG]2020-06-20 05:30:34,349:utils:loss_avg: 1.06543, lr_pg0:0.006683439175686144, lr_pg1: 0.006683439175686144final_score:0.52044, mc_score:0.02184
[DEBUG]2020-06-20 05:30:34,722:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:34,726:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:35,026:utils:on_backward_begin lr: 0.007079457843841376
[DEBUG]2020-06-20 05:30:35,027:utils:itr: 194, num_batch: 194, last loss: 2.797308, smooth_loss: 1.185721
[DEBUG]2020-06-20 05:30:35,037:utils:loss_avg: 1.07431, lr_pg0:0.007079457843841376, lr_pg1: 0.007079457843841376final_score:0.51598, mc_score:0.01723
[DEBUG]2020-06-20 05:30:35,406:utils:grad info pg0: norm std(0.000000) mean(0.000001)
[DEBUG]2020-06-20 05:30:35,409:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:35,709:utils:on_backward_begin lr: 0.007498942093324556
[DEBUG]2020-06-20 05:30:35,711:utils:itr: 195, num_batch: 195, last loss: 1.827399, smooth_loss: 1.198804
[DEBUG]2020-06-20 05:30:35,721:utils:loss_avg: 1.07815, lr_pg0:0.007498942093324556, lr_pg1: 0.007498942093324556final_score:0.51684, mc_score:0.01837
[DEBUG]2020-06-20 05:30:36,089:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:36,092:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:36,389:utils:on_backward_begin lr: 0.007943282347242814
[DEBUG]2020-06-20 05:30:36,390:utils:itr: 196, num_batch: 196, last loss: 1.754774, smooth_loss: 1.210135
[DEBUG]2020-06-20 05:30:36,402:utils:loss_avg: 1.08159, lr_pg0:0.007943282347242814, lr_pg1: 0.007943282347242814final_score:0.51650, mc_score:0.01810
[DEBUG]2020-06-20 05:30:36,770:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:36,773:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:37,073:utils:on_backward_begin lr: 0.008413951416451949
[DEBUG]2020-06-20 05:30:37,075:utils:itr: 197, num_batch: 197, last loss: 7.597126, smooth_loss: 1.340258
[DEBUG]2020-06-20 05:30:37,087:utils:loss_avg: 1.11450, lr_pg0:0.008413951416451949, lr_pg1: 0.008413951416451949final_score:0.51522, mc_score:0.01700
[DEBUG]2020-06-20 05:30:37,462:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:37,466:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:37,782:utils:on_backward_begin lr: 0.008912509381337454
[DEBUG]2020-06-20 05:30:37,783:utils:itr: 198, num_batch: 198, last loss: 1.786048, smooth_loss: 1.349337
[DEBUG]2020-06-20 05:30:37,793:utils:loss_avg: 1.11787, lr_pg0:0.008912509381337454, lr_pg1: 0.008912509381337454final_score:0.51394, mc_score:0.01536
[DEBUG]2020-06-20 05:30:38,161:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:38,164:utils:grad info pg1: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:38,461:utils:on_backward_begin lr: 0.009440608762859232
[DEBUG]2020-06-20 05:30:38,462:utils:itr: 199, num_batch: 199, last loss: 1.328048, smooth_loss: 1.348903
[DEBUG]2020-06-20 05:30:38,473:utils:loss_avg: 1.11892, lr_pg0:0.009440608762859232, lr_pg1: 0.009440608762859232final_score:0.51241, mc_score:0.01368
[DEBUG]2020-06-20 05:30:38,842:utils:grad info pg0: norm std(0.000000) mean(0.000000)
[DEBUG]2020-06-20 05:30:38,845:utils:grad info pg1: norm std(0.000000) mean(0.000000)
```

## Version 28
0.9373
## Version 29
0.9410 with 3e-06 learning rate

Q: different lr, the final negative predicted value changed.
