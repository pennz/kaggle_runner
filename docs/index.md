[![Build Status](https://travis-ci.org/pennz/kaggle_runner.svg?branch=master)](https://travis-ci.org/pennz/kaggle_runner)
[![PyPI version](https://badge.fury.io/py/kaggle-runner.svg)](https://badge.fury.io/py/kaggle-runner)
[![Maintainability](https://api.codeclimate.com/v1/badges/979bc98e4acb59a5e1aa/maintainability)](https://codeclimate.com/github/pennz/kaggle_runner/maintainability)
[![pipeline status](https://gitlab.com/MrCue/kaggle_runner/badges/master/pipeline.svg)](https://gitlab.com/MrCue/kaggle_runner/-/commits/master)
[![coverage report](https://gitlab.com/MrCue/kaggle_runner/badges/master/coverage.svg)](https://gitlab.com/MrCue/kaggle_runner/-/commits/master)

[Generated doc website](https://pennz.github.io/kaggle_runner)

# kaggle_runner

Check main.py or test/test_coord.py for usage. It uses kaggle API to upload your script/notebook to kaggle servers and let the kernel run. And you will get running logs through message queue.

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

A demo:

1. \#0 Left panel: tcpserver listen for reverse shells
1. \#1 Upper panel: Logs from interactive session to our tcpserver which receive logs
1. \#2 Second upper panel: AMQP logs received
1. \#3 Main panel: vim window
1. \#4 Right bottom panel: logged in reverse shell for commit session

[![asciicast](https://asciinema.org/a/vcLKH8MEkxv4WYEb9xxK8xBnU.svg)](https://asciinema.org/a/vcLKH8MEkxv4WYEb9xxK8xBnU)
