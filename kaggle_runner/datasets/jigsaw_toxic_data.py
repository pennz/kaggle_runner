#!/bin/env python
import os
import pandas as pd
from kaggle_runner import logger
import random

DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification"
TRD = "jigsaw-toxic-comment-train.csv"
VD = "validation.csv"
TD = "test.csv"
OUT_PATH = "/tmp/input.txt"


def get_column(csv_file, col_name):
    d = pd.read_csv(csv_file)

    return d[col_name]
def get_toxic_comment(p):
    dtr = get_column(os.path.join(DATA_PATH, p), "comment_text")

    r = random.randint(0, len(dtr)-1)
    logger.debug("toxic comment data example: %s", dtr[r])

    return dtr

def merge_all_data():
    dtr = get_toxic_comment(TRD)
    vd = get_toxic_comment(VD)
    td = get_toxic_comment(TD)
    c = pd.concat([dtr, vd, td])
    pd.to_csv(OUT_PATH, index=False)
    print("%d %d %d" % (len(dtr), len(vd), len(td)))

if __name__ == "__main__":
    merge_all_data()
