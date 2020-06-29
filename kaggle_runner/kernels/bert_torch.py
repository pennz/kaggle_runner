from __future__ import absolute_import, division, print_function
import sys
package_dir_a = "/kaggle/input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from kaggle_runner.datasets.bert import DATA_PATH, BERT_BASE_DIR, PRETRAIND_PICKLE_AND_MORE
from kaggle_runner import may_debug, logger
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import roc_auc_score
from sklearn import metrics, model_selection
from pytorch_pretrained_bert import (BertAdam, BertForSequenceClassification,
                                     BertTokenizer, BertConfig,
                                     convert_tf_checkpoint_to_pytorch)  # needed fused_layer_norm_cuda, so TPU won't work

import datetime
import gc
import operator
import os
import pickle
import re
import shutil
import time
import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pkg_resources
import scipy.stats as stats
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from IPython.core.interactiveshell import InteractiveShell
from nltk.stem import PorterStemmer


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings(action='once')


EPOCHS = 3
WORK_DIR = '/kaggle/working/'


def prepare_pretrained():
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        BERT_BASE_DIR + '/bert_model.ckpt',
        BERT_BASE_DIR + '/bert_config.json',
        WORK_DIR + 'pytorch_model.bin')

    shutil.copyfile(BERT_BASE_DIR + '/bert_config.json',
                    WORK_DIR + 'bert_config.json')
# -


def get_trained_model(fine_tuned = "bert_pytorch.bin", device=torch.device('cuda')):
    model = None
    y_columns = ['toxic', "severe_toxic","obscene","threat","insult","identity_hate"]
    pretrain_data_folder = PRETRAIND_PICKLE_AND_MORE

    if not os.path.exists(pretrain_data_folder+"/" + fine_tuned):
        pretrain_data_folder = '/home/working'

    if os.path.exists(pretrain_data_folder+"/"+fine_tuned):
        output_model_file = pretrain_data_folder+"/"+fine_tuned
        bert_config = BertConfig.from_json_file(pretrain_data_folder + "/bert_config.json")

        # Run validation
        # The following 2 lines are not needed but show how to download the model for prediction
        model = BertForSequenceClassification(
            bert_config, num_labels=len(y_columns))
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

    return model

def get_predict(model, test_dataset, device=torch.device('cuda')):
    if model is None:
        return

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    test_preds = np.zeros((len(test_dataset)))
    test = torch.utils.data.TensorDataset(
        torch.tensor(test_dataset, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=32, shuffle=False)

    tk0 = tqdm(test_loader)

    for i, (x_batch,) in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(
            x_batch > 0).to(device), labels=None)
        test_preds[i*32:(i+1)*32] = pred[:, 0].detach().cpu().squeeze().numpy()

    return test_preds

def get_validation_result(model, test_dataset, y_targets, device=torch.device('cuda')) :
    test_preds = get_predict(model, test_dataset, device)
    pred = 1 / (1+np.exp(- test_preds))

    if len(y_targets.shape) == 1:
        return roc_auc_score(y_targets, pred)

    return roc_auc_score(y_targets[:,0], pred)

def get_test_result(self, test_dataset, device=torch.device('cuda'), data_path=DATA_PATH):
    test_preds = get_predict(self.model, test_dataset, device)
    pred = 1 / (1+np.exp(- test_preds))

    sub = pd.read_csv(os.path.join(data_path , 'sample_submission.csv'))
    sub['toxic'][:len(pred)] = pred
    sub.to_csv('submission.csv', index=False)


def for_pytorch(data_package, device=torch.device('cuda'), SEED=118, phase="predict", model=None):

    if device is None and os.getenv("TPU_NAME") is not None:
        import torch_xla # model
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

    X, y, X_val, y_val, X_test = data_package

    if model is None:
        try:
            model = get_trained_model(device=device)
        except RuntimeError as e:
            logger.debug("%s", e)

    if model is not None and phase=="predict":
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        valid_preds = np.zeros((len(X_val)))
        valid = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.long))
        valid_loader = torch.utils.data.DataLoader(
            valid, batch_size=32, shuffle=False)

        tk0 = tqdm(valid_loader)

        for i, (x_batch,) in enumerate(tk0):
            pred = model(x_batch.to(device), attention_mask=(
                x_batch > 0).to(device), labels=None)
            valid_preds[i*32:(i+1)*32] = pred[:, 0].detach().cpu().squeeze().numpy()
    else:
        import subprocess
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(
            X, dtype=torch.long), torch.tensor(y, dtype=torch.float))
        output_model_file = "bert_pytorch.bin"

        lr = 1e-5
        batch_size = 32
        accumulation_steps = 3
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False

        if model is None:
            prepare_pretrained()
            model = BertForSequenceClassification.from_pretrained(
                ".", cache_dir=None, num_labels=1 if len(y[0]) < 1 else len(y[0]))
            assert model is not None
        logger.info("AUC for valication: %f", get_validation_result(model, X_val, y_val))
        model.zero_grad()
        model = model.to(device)

        param_optimizer = list(model.named_parameters())
        may_debug()

        req_grad = ['layer.10', 'layer.11', 'bert.poole', 'classifier']
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        def para_opt_configure(req_grad, no_decay):
            for n, p in param_optimizer:
                if any(nd in n for nd in req_grad):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            return optimizer_grouped_parameters
        optimizer_grouped_parameters = para_opt_configure(req_grad, no_decay)
        train = train_dataset

        num_train_optimization_steps = int(
            EPOCHS*len(train)/batch_size/accumulation_steps)

        optimizer = BertAdam(optimizer_grouped_parameters,
                       lr=lr,
                       warmup=0.05,
                       t_total=num_train_optimization_steps)

        subprocess.run('python3 -m pip show apex || ([ -d /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a ] && '
            'pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a)',
                       shell=True, check=True)
        from apex import amp  # automatic mix precision
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", verbosity=1)
        model = model.train()

        tq = tqdm(range(EPOCHS))

        for epoch in tq:
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, shuffle=True)
            avg_loss = 0.
            avg_accuracy = 0.
            lossf = None
            para_opt_configure(req_grad, no_decay)  # valication will change it
            tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            optimizer.zero_grad()   # Bug fix - thanks to @chinhuic

            for i, (x_batch, y_batch) in tk0:
                #        optimizer.zero_grad()
                y_pred = model(x_batch.to(device), attention_mask=(
                    x_batch > 0).to(device), labels=None)
                loss = F.binary_cross_entropy_with_logits(
                    y_pred, y_batch.to(device))
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                    optimizer.step()                            # Now we can do an optimizer step
                    optimizer.zero_grad()

                if lossf:
                    lossf = 0.98*lossf+0.02*loss.item()
                else:
                    lossf = loss.item()
                tk0.set_postfix(loss=lossf)
                avg_loss += loss.item() / len(train_loader)
                avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:, 0]) > 0.5) == (
                    y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item()/len(train_loader)
            tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)
            logger.info("AUC for valication: %f", get_validation_result(model, X_val, y_val))

        from datetime import date
        today = date.today()
        torch.save(model.state_dict(), f"{today}_{output_model_file}")
# +
## +
## From baseline kernel
#
#    def calculate_overall_auc(df, model_name):
#        true_labels = df[TOXICITY_COLUMN] > 0.5
#        predicted_labels = df[model_name]
#
#        return metrics.roc_auc_score(true_labels, predicted_labels)
#
#    def power_mean(series, p):
#        total = sum(np.power(series, p))
#
#        return np.power(total / len(series), 1 / p)
#
#    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
#        bias_score = np.average([
#            power_mean(bias_df[SUBGROUP_AUC], POWER),
#            power_mean(bias_df[BPSN_AUC], POWER),
#            power_mean(bias_df[BNSP_AUC], POWER)
#        ])
#
#        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
#
#    SUBGROUP_AUC = 'subgroup_auc'
#    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
#    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
#
#    def compute_auc(y_true, y_pred):
#        try:
#            return metrics.roc_auc_score(y_true, y_pred)
#        except ValueError:
#            return np.nan
#
#    def compute_subgroup_auc(df, subgroup, label, model_name):
#        subgroup_examples = df[df[subgroup] > 0.5]
#
#        return compute_auc((subgroup_examples[label] > 0.5), subgroup_examples[model_name])
#
#    def compute_bpsn_auc(df, subgroup, label, model_name):
#        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
#        subgroup_negative_examples = df[(
#            df[subgroup] > 0.5) & (df[label] <= 0.5)]
#        non_subgroup_positive_examples = df[(
#            df[subgroup] <= 0.5) & (df[label] > 0.5)]
#        examples = subgroup_negative_examples.append(
#            non_subgroup_positive_examples)
#
#        return compute_auc(examples[label] > 0.5, examples[model_name])
#
#    def compute_bnsp_auc(df, subgroup, label, model_name):
#        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
#        subgroup_positive_examples = df[(
#            df[subgroup] > 0.5) & (df[label] > 0.5)]
#        non_subgroup_negative_examples = df[(
#            df[subgroup] <= 0.5) & (df[label] <= 0.5)]
#        examples = subgroup_positive_examples.append(
#            non_subgroup_negative_examples)
#
#        return compute_auc(examples[label] > 0.5, examples[model_name])
#
#    def compute_bias_metrics_for_model(dataset,
#                                       subgroups,
#                                       model,
#                                       label_col,
#                                       include_asegs=False):
#        """Computes per-subgroup metrics for all subgroups and one model."""
#        records = []
#
#        for subgroup in subgroups:
#            record = {
#                'subgroup': subgroup,
#                'subgroup_size': len(dataset[dataset[subgroup] > 0.5])
#            }
#            record[SUBGROUP_AUC] = compute_subgroup_auc(
#                dataset, subgroup, label_col, model)
#            record[BPSN_AUC] = compute_bpsn_auc(
#                dataset, subgroup, label_col, model)
#            record[BNSP_AUC] = compute_bnsp_auc(
#                dataset, subgroup, label_col, model)
#            records.append(record)
#
#        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
#
#
## +
#
#    test_df = train_df.tail(valid_size).copy()
#    train_df = train_df.head(num_to_load)
#    MODEL_NAME = 'model1'
#    test_df[MODEL_NAME] = torch.sigmoid(torch.tensor(valid_preds)).numpy()
#    TOXICITY_COLUMN = 'target'
#    bias_metrics_df
#    get_final_metric(
#        bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))
## -
