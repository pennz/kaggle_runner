import os
import subprocess

SEED = 1

FOCAL_LOSS_GAMMA = 0.0
FOCAL_LOSS_GAMMA_NEG_POS = 0.25
FOCAL_LOSS_BETA_NEG_POS = 1.0
# ALPHA = 0.91
ALPHA = 0.5

CONVERT_DATA = False
CONVERT_DATA_Y_NOT_BINARY = "convert_data_y_not_binary"
# given the pickle of numpy train data
CONVERT_TRAIN_DATA = "convert_train_data"
# given the pickle of numpy train data
CONVERT_ADDITIONAL_NONTOXIC_DATA = "CONVERT_ADDITIONAL_NONTOXIC_DATA"

EXCLUDE_IDENTITY_IN_TRAIN = True
TRAIN_DATA_EXCLUDE_IDENDITY_ONES = "TRAIN_DATA_EXCLUDE_IDENDITY_ONES"
DATA_ACTION_NO_NEED_LOAD_EMB_M = "DATA_ACTION_NO_NEED_LOAD_EMB_M"

NEG_RATIO = 1 - 0.05897253769515213
Data_Folder = '/home/'
LOAD_BERT_DATA = False

RIPDB = os.environ.get("RIPDB") == 'true'
DEBUG = os.environ.get("DEBUG") == 'true'
INTERACTIVE = False
try:
	__note = subprocess.check_output('pgrep -f "jupyter notebook"', shell=True)

	if __note.strip() != "":
		INTERACTIVE = True
except:
	pass
