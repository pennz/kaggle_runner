#!/bin/env python
import csv
import os
import random
import re
import subprocess

import pandas as pd

from kaggle_runner import logger

DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification"
TRD = "jigsaw-toxic-comment-train.csv"
VD = "validation.csv"
TD = "test.csv"
OUT_PATH = "/tmp/input.txt"


def get_column(csv_file, col_name):
    d = pd.read_csv(csv_file)

    return d[col_name]

def get_toxic_comment(p, col_name="comment_text"):
    dtr = get_column(os.path.join(DATA_PATH, p), col_name=col_name)

    r = random.randint(0, len(dtr)-1)
    logger.debug("toxic data example %s.[%s]:\n %s\n", p, col_name, dtr[r])

    return dtr

def join_line(ml):
    return ml.replace("\n", "\\n")

def clean(text):
    text = text.fillna("fillna")
    text = text.map(lambda x: re.sub('\\n', ' ', str(x)))
    text = text.map(lambda x: re.sub(
        "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))
    text = text.map(lambda x: re.sub(
        "\(http://.*?\s\(http://.*\)", '', str(x)))

    return text

def merge_all_data():
    dtr = get_toxic_comment(TRD)
    vd = get_toxic_comment(VD)
    td = get_toxic_comment(TD, "content")
    dtr =clean(dtr)
    vd = clean( vd)
    td = clean( td)
    c = pd.concat([dtr, vd, td])
    pd.DataFrame.to_csv(c,OUT_PATH, index=False, header=False)
    print("%d %d %d, %d" % (len(dtr), len(vd), len(td), sum((len(dtr), len(vd), len(td))))) # lines info: 223549 8000 63812


if __name__ == "__main__":
    merge_all_data()
