[![Build Status](https://travis-ci.org/pennz/kaggle_runner.svg?branch=master)](https://travis-ci.org/pennz/kaggle_runner)
[![PyPI version](https://badge.fury.io/py/kaggle-runner.svg)](https://badge.fury.io/py/kaggle-runner)
[![Maintainability](https://api.codeclimate.com/v1/badges/979bc98e4acb59a5e1aa/maintainability)](https://codeclimate.com/github/pennz/kaggle_runner/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/979bc98e4acb59a5e1aa/test_coverage)](https://codeclimate.com/github/pennz/kaggle_runner/test_coverage)

# kaggle_runner

Check main.py or test/test_coord.py for usage. It uses kaggle API to upload your script/notbook to kaggle servers and let the kernel run. And you will get running logs througn message queue.

## AMQP
AMQP is used for logging. Its license needs mention.

## Example
Use this [kaggle competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) about Pneumothorax Segmentation as an example. To run the example, you will need a kaggle account and set the kaggle command line tool up. And you issues this command to let it run:

```sh
# install kaggle_runner, which will pull kaggle command line tool as the dependency
pip install kaggle_runner

# put your kaggle API token to the right place
cat > ~/.kaggle/kaggle.json <<EOF
{
  "username": "YOUR_KAGGLE_USER_NAME",
  "key": "YOUR_KAGGLE_API_ACCESS_TOKEN",
}
EOF

# kaggle_runner will use kaggle API to push the template kernel codes to kaggle server and wait message back
python -m kaggle_runner
```
