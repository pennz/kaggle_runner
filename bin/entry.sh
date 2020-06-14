#!/bin/bash
PS4='Line ${LINENO}: ' bash -x runner.sh pennz kaggle_runner master "dev" 1 vtool.duckdns.org "23454" "amqp://kaggle:9b83ca70cf4cda89524d2283a4d675f6@pengyuzhou.com/" "384" "19999" "intercept-resnet" | tee runner_log
