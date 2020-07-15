.DEFAULT_GOAL := help

#export LD_LIBRARY_PATH := $(PWD)/lib:$(LD_LIBRARY_PATH)
export PATH := /nix/store/3ycgq0lva60yc2bw4qshmlsaqn0g90x4-nodejs-14.2.0/bin:$(HOME)/.local/bin:$(PWD)/bin:$(PATH)
export DEBUG := $(DEBUG)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43 

PY_SRC := src/ tests/ scripts/
CI ?= false
TESTING ?= false

JUPYTER_PARAMS := --NotebookApp.token=greatday --NotebookApp.notebook_dir=/content/ --NotebookApp.allow_origin=* --NotebookApp.disable_check_xsrf=True --NotebookApp.iopub_data_rate_limit=10010000000 --NotebookApp.open_browser=False --allow-root
TOXIC_DEP := coverage ipdb pyicu pycld2 polyglot textstat googletrans transformers==2.5.1 pandarallel catalyst==20.4.2 colorama parse pysnooper ripdb pytest-logger python_logging_rabbitmq

define _write_dataset_list
cat >.datasets <<'EOF'
"gabrichy/nvidiaapex",
"matsuik/ppbert", #for pytorch pretrained bert
"maxjeblick/bert-pretrained-models", #for pytorch pretrained bert
"k1gaggle/clean-pickle-for-jigsaw-toxicity", # XLMRobert(sh) and XLNET data
"k1gaggle/jigsaw-multilingula-toxicity-token-encoded",
"shonenkov/open-subtitles-toxic-pseudo-labeling",
"k1gaggle/jigsaw-toxicity-train-data-with-aux",
"shonenkov/jigsaw-public-baseline-train-data",
"shonenkov/jigsaw-public-baseline-results",
"kashnitsky/jigsaw-multilingual-toxic-test-translated",
"pranshu29/jigsaw-new-balanced-dataset"
EOF
endef
export write_dataset_list_script = $(value _write_dataset_list)

UNBUFFER := $(shell which unbuffer)
ifneq ($(UNBUFFER),)
	UNBUFFERP := $(UNBUFFER) -p
endif

KAGGLE_USER_NAME=$(shell jq -r '.username' ~/.kaggle/kaggle.json)
KIP=$(shell ip addr show dev eth0 | grep inet | sed 's/.*inet \([^\/]*\).*/\1/')


SED := $(shell which gsed &>/dev/null)
ifneq ($(SED),)
	SED := gsed
else
	SED := sed
endif
export SED := $(SED)

SERVER := $(SERVER)
CHECK_PORT := $(CHECK_PORT)
ifeq ($(SERVER),)
	SERVER := pengyuzhou.com
endif
ifeq ($(CHECK_PORT),)
	CHECK_PORT := 23455
endif

export SERVER
export CHECK_PORT

URL="https://www.kaggleusercontent.com/kf/33961266/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..b3ZzhVJx_c1vhjL3vVc5Ow.4i-Vpk1-bF9zCZJP7LHiuSY44ljoCyKbD7rLcvDSUuViAHL3Xw_Idb3gkMIGhqY6kLN9GX2VzGdxAv9qqOJGXYc7EUeljbX6dvjdssk5Iuhwl4kxz-TIsWYaxqONbMGBQX9rT-nIJYmpjV8UKle7DlX1UYFJKhLYyuckV1B5ZEGHkRjdzwasPlhc8IJkX83RfLhe7C6T0pR8oFU-gmvtQxSvKzXprbYvPQVRMyBf4xD8Bm9xvEq8aFVIiwHGROwvIcorUhZ3cHsCXRSE6RDm7f1rmbA_52xetuCEB2de1_tg-XZ7FoBx6_QaQHXnZWWRhZ1Edyzt5LlakbQI55Ncq3RBByr84QnJmAc9yJORqorQrtEWuAXCrHbYTiKR39i4sm2mkcvIhdgqYuHh8E7ZMXt7MiYr4W6Na233NBRPzY4l15DXqV5ZXp_m-th1ljwxUK8AvNTo0Qs3PNd0bvezFQew10jrMR-N-Z8ZFqtX--Ba8BbMFex6_jJxhN6JXFOXPwCJUWhrZ1yYNE3iqpavJkOM06Vkx6UEOhNbawmPrDtzF4vXViCdHbfUTcpd2qvmXgVlTg7cULSw4MzGdN-Uqbp6-MnpvGIFrRVOVooRE5u8zhrbRcZL4RApjr9SrIEPm1WSp7Qlj8wjktBL4K1bNKn4NE9-AFtOu_0X-lL0Afav41RxxhqQyL_Ox3o3YI8Y.hz022ycDLUciahf-YOeEDw/inceptionresnetv2-520b38e4.pth"
PY=poetry run python3
SRC=$(wildcard */**.py)
SHELL=/bin/bash


IS_CENTOS=type firewall-cmd >/dev/null 2>&1

_: test
	@echo "DONE $@"

.PHONY: test
test: ctr ## Main test function, run coverage test.
	@echo "DONE $@"

.PHONY: test_bert_torch
test_bert_torch: pytest ## Test bert written in pytorch.
	if [ -z $$DEBUG ]; then $(PY) tests/test_bert_torch.py 2>&1 | $(UNBUFFERP) tee -a test_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT); \
else wt $(PY) -m pdb tests/test_bert_torch.py </dev/tty ; fi

.PHONY: pytest
pytest: ## Install pytest.
	$(PY) -m pip show pytest | grep "Version: 5." &>/dev/null || ($(PY) -m pip install --upgrade pytest && $(PY) -m pip install --upgrade pytest-cov)

.PHONY: check_log_receiver
check_log_receiver: ## Check log receiver in CHECK_PORT.
	@echo "$@" will use tcp to receive logs
	-pkill -f "$(CHECK_PORT)"
	-$(IS_CENTOS) && (pgrep -f firewalld >/dev/null || sudo systemctl start firewalld)
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp --permanent
	ncat -vkl --recv-only  -p $(CHECK_PORT) -o logs_check & sleep 1; tail -f logs_check # logs_check will be used by pcc to get mosh-client connect authentication info

.PHONY: pc
pc: ## Connect to reverse shell.
	pcc
	make connect

.PHONY: m
m: ## Mosh server too.
	while true; do (setup_mosh_server 2>&1 | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT)) & sleep $$((60*25)); done

.PHONY: mosh
mosh: ## Mosh server in while loop.
	( while true; do bash -x setup_mosh_server& [ -f /tmp/mexit ] && exit 0; sleep 600; done 2>&1 | $(UNBUFFERP) tee -a ms_connect_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT) ) &
	#@sleep 1
	#tail ms_connect_log

.PHONY: rvs_session
rvs_session: ## Create new tmux session for reverse shell.
	-tmux new-session -d -n "good-day" -s rvsConnector "cat"
	-tmux set-option -t rvsConnector renumber-windows on

.PHONY: _pccnct
_pccnct:
	-pkill -f "50001.*addNew"
	echo "start mosh connector";
	$(UNBUFFER) ncat -uklp 50001 -c "bash -c 'echo $$(date): New Incoming >>mosh_log'; echo; addNewNode.sh mosh" &
	echo "connection listener setup done."
	echo "pccnct has been put to backgound."
	
.PHONY: pccnct
pccnct: rvs_session _pccnct ## Setup reverse shell receiver
	make check_log_receiver & # will output to current process
	-$(IS_CENTOS) && sudo service rabbitmq-server start # For AMQP log, our server 
	@echo "pc connector started now"

.PHONY: ctr
ctr: kr check install_dep pytest $(SRC) ## Coverage Test Report
	-timeout 10 git push
	[ -f bin/cc-test-reporter ] || curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > bin/cc-test-reporter
	chmod +x bin/cc-test-reporter
	-bin/cc-test-reporter before-build
	-$(PY) -m coverage run -m pytest -vs --full-trace tests
	-$(PY) -m coverage report -m -i | grep '^TOTAL.*[0-9]\{1,\}'
	-$(PY) -m coverage xml -i -o coverage.xml
	-bin/cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT

.PHONY: get_submission
get_submission: ## Get submission from kaggle.
	kaggle datasets download --file submission.csv --unzip k1gaggle/bert-for-toxic-classfication-trained
	-unzip '*.zip' && rm *.zip && mv *.csv submission.csv

.PHONY: push
push: rvs_session $(SRC) ## Push code to kernel.
	-#git push # push first as kernel will download the codes, so put new code to github first
	-@echo "$$(which $(PY)) is our $(PY) executable"; [[ x$$(which $(PY)) =~ conda ]]
	sed -i 's/\(id": "\)\(.*\)\//\1$(KAGGLE_USER_NAME)\//' kaggle_runner/runner_template/kernel-metadata.json
	title=$$(git show --no-patch --oneline | tr " " "_"); sed -i 's/title\(.*\)|.*"/title\1| '$$title\"/ kaggle_runner/runner_template/kernel-metadata.json
	git add kaggle_runner/runner_template/kernel-metadata.json && git commit -sm "Update metadata when push to server" --no-gpg && git push &
	run_coordinator $(PHASE) # source only works in specific shell: bash or ...

.PHONY: connect
connect: ## Connect to mosh server.
	tmux select-window -t rvsConnector:{end}
	tmux switch -t rvsConnector:{end}

.PHONY: inner_lstm
inner_lstm:
	while true; do test x$$(git pull | grep -c Already) = x1 || { $(PY) \
lstm.py 2>&1 | tee -a lstm_log; };  echo "$$(date) $$HOSTNAME CPU: "$$(grep \
'cpu ' /proc/stat >/dev/null;sleep 0.1; grep 'cpu ' /proc/stat | awk -v RS='' \
'{print ($$13-$$2+$$15-$$4)*100/($$13-$$2+$$15-$$4+$$16-$$5)}')% 'Mem: '$$(awk \
'/MemTotal/{t=$$2}/MemAvailable/{a=$$2}END{print 100-100*a/t}' /proc/meminfo)% \
.PHONY: 'Uptime
'Uptime: '$$(uptime | awk '{print $$3}'); sleep 10; done ## 'Uptime

.PHONY: lstm
lstm: ## Run lstm.py in loop.
	-pkill -f "inner_lstm"
	make inner_lstm

.PHONY: debug_toxic
debug_toxic: ## Debug toxic code.
	DEBUG=true make toxic #python3 -m pdb $$(which pytest) -sv tests/test_distilbert_model.py

.PHONY: toxic
toxic: check install_dep ## Run toxic code (Not Debug).
	echo $$(ps aux | grep "make $@$$")
	echo "$@:" DEBUG flag is $$DEBUG .
	bash -c 'ppid=$$PPID; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); while [[ -n "$$mpid" ]] && [[ "$$mpid" -lt "$$((ppid-10))" ]]; do if [ ! -z $$mpid ]; then echo "we will kill existing \"make $@\" with pid $$mpid"; kill -9 $$mpid; sleep 1; else return 0; fi; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); done'
	if [ -z $$DEBUG ]; then $(UNBUFFER) $(PY) tests/test_distilbert_model.py 2>&1 | $(UNBUFFERP) tee -a toxic_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT); else wt '$(PY) -m ipdb tests/test_distilbert_model.py'; fi
	-git stash pop || true

.PHONY: test_coor
test_coor: update_code $(SRC) ## Test coordinator.
	$(PY) -m pytest -k "test_generate_runner" tests/test_coord.py; cd .runners/intercept-resnet-384/ && $(PY) main.py

.PHONY: clean
clean: _clean ## Clean.
	#-bash -c 'currentPpid=$$(pstree -spa $$$$ | $(SED) -n "2,3 p" |  cut -d"," -f 2 | cut -d" " -f 1); pgrep -f "rvs.sh" | sort | grep -v -e $$(echo $$currentPpid | $(SED) "s/\s\{1,\}/ -e /" ) -e $$$$ | xargs -I{} kill -9 {}'
	-ps aux | grep "ncat .*lp" | grep -v "while" | grep -v "50001" | grep -v "grep" | tee /dev/tty | awk '{print $$2} ' | xargs -I{} kill -9 {}
	-rm -rf __pycache__ mylogs dist/* build/*

_clean:  ## Delete temporary files.
	-@rm -rf build 2>/dev/null
	-@rm -rf .coverage* 2>/dev/null
	-@rm -rf dist 2>/dev/null
	-@rm -rf .mypy_cache 2>/dev/null
	-@rm -rf pip-wheel-metadata 2>/dev/null
	-@rm -rf .pytest_cache 2>/dev/null
	-@rm -rf src/*.egg-info 2>/dev/null
	-@rm -rf src/mkdocstrings/__pycache__ 2>/dev/null
	-@rm -rf scripts/__pycache__ 2>/dev/null
	-@rm -rf site 2>/dev/null
	-@rm -rf tests/__pycache__ 2>/dev/null
	-@find . -name "*.rej" -delete 2>/dev/null


.PHONY: submit
submit: ## submit
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

.PHONY: twine
_twine:
	@$(PY) -m twine -h >/dev/null || ( echo "twine not found, will install it." ; $(PY) -m pip install --user --upgrade twine )

.PHONY: publish
publish: _twine ## Publish to pip.
	if [[ x$(TAG) =~ xv ]] || [ -z $(TAG) ]; then >&2 echo "Please pass TAG \
flag when you call make, and use something like 0.0.3, not v0.0.3"; false; else \
gsed -i 's/version=.*/version=\"$(TAG)\",/' setup.py || \
$(SED) -i 's/version=.*/version=\"$(TAG)\",/' setup.py ;\
git add setup.py && \
(git tag -d "v$(TAG)"; git push --delete origin "v$(TAG)" || true) && \
git commit -sm "setup.py: v$(TAG)" && (git tag -s "v$(TAG)" || true) && git push --tags; \
fi
	$(PY) setup.py sdist bdist_wheel
	$(PY) -m twine upload dist/*

.PHONY: install_dep_seg
install_dep_seg: ## Install dependency about segmentation.
	bash -c '(test -z "$$($(PY) -m albumentations 2>&1 | grep direct)" && $(PY) -m pip install -U git+https://github.com/albu/albumentations) & \
(test -z "$$($(PY) -m segmentation_models_pytorch 2>&1 | grep direct)" && $(PY) -m pip install git+https://github.com/qubvel/segmentation_models.pytorch) & \
wait'


.PHONY: $(TOXIC_DEP)
$(TOXIC_DEP): ## Install pip dependencies if not installed.
	@echo "Installing $@"
	#$(PY) -m pip show $@ &>/dev/null || $(PY) -m pip install -q $@

.PHONY: install_dep
install_dep: $(TOXIC_DEP) pytest ## install_dep
	for p in $^; do ($(PY) -m pip show $$p &>/dev/null || $(PY) -m pip install -q $$p) & done; wait
	#$(PY) -m pip install -q eumetsat expect &
	#conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago
	@echo make $@ $^ done

.PHONY: rpdbrvs
rpdbrvs: ## rpdbrvs
	while true; do ncat $(SERVER) 23454 --sh-exec 'ncat -w 3 127.1 4444; echo \# nc return $?' ; sleep 1; echo -n "." ; done;

.PHONY: rvs
rvs: ## rvs
	while true; do ncat -w 60s -i 1800s $(SERVER) $$(( $(CHECK_PORT) - 1 )) -c "echo $$(date) \
started connection; echo $$HOSTNAME; python -c 'import pty; \
pty.spawn([\"/bin/bash\", \"-li\"])'"; echo "End of one rvs."; sleep 5; done;

.PHONY: r
r: ## r
	bash -c "SAVED_STTY=$$(stty -g); stty raw -echo; while true; do ncat -v $(SERVER) $$(( $(CHECK_PORT) - 1 )); echo DONE connect to reverse shell.; echo RET: $$?.; sleep 3; done; stty $$SAVED_STTY"

.PHONY: dbroker
dbroker: ## Debug broker setup.
	while true; do set -x; echo "Start Listening"; ncat --broker -v -m 2 -p $$(( $(CHECK_PORT) - 1 )); echo >&2 "Listen failed, will restart again." ; sleep 5; done & # just one debug session at a time, more will make you confused

.PHONY: broker
broker: dbroker r ## Broker and listen to broker.
	@echo "Done broker for remote debug."

.PHONY: rpdbc
rpdbc: ## rpdbc
	bash -c "SAVED_STTY=$$(stty -g); stty onlcr onlret -icanon opost -echo -echoe -echok -echoctl -echoke; ncat -v 127.0.0.1 23454; stty $$SAVED_STTY"

.PHONY: mq
mq: ## Message queue.
	make amqp_log &
	id -u rabbitmq &>/dev/null; \
if [ $$? -eq 0 ]; then \
while [ $$(ps -u rabbitmq | wc -l) -lt 5 ]; do \
  sleep 60; ps aux | grep "amqp" | tee /dev/tty | grep -v -e "sh" -e "grep" | \
  awk '{print $$2} ' | xargs -I{} kill {}; \
  make amqp_log & \
  echo "You can check background jobs:"; jobs; \
done; \
fi

.PHONY: amqp_log
amqp_log: ## Receive AMQP log.
	-$(IS_CENTOS) && sudo systemctl restart rabbitmq-server.service
	$(UNBUFFER) receive_logs_topic \*.\* 2>&1 | $(UNBUFFERP) tee -a mq_log | $(UNBUFFERP) $(SED) -n 's/^.*\[x\] \(.*\)/\1/p'  | (type jq >/dev/null 2>&1 && $(UNBUFFERP) jq -r '.msg' || $(UNBUFFERP) cat -)
	# sleep 3; tail -f mq_log | $(SED) -n "s/\(.*\)\[x.*/\1/p"

.PHONY: mlocal
mlocal: ## Set mosh session window size.
	tty_config=$$(stty -g); size=$$(stty size); $(MC); stty $$tty_config; stty columns $$(echo $$size | cut -d" " -f 2) rows $$(echo $$size | cut -d" " -f 1)

.PHONY: check
check: ## Check.
	-ps aux | grep make
	echo [sed] use $(SED).
	echo PATH $(PATH)
	-@echo $(http_proxy)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT) | ncat $(SERVER) $(CHECK_PORT)
	-expect -h
	pstree -laps $$$$
	-poetry run echo "$$(which python3) is our python executable"; \
$(PY) -c 'import sys; print(sys.path);'; \
if [[ x$$(which $(PY)) =~ conda ]]; then echo conda env fine; else echo >&2 conda env not set correctly, please check.; source ~/.bashrc; conda activate pyt; fi
	@$(PY) -c 'import os; print("$@: DEBUG=%s" % os.environ.get("DEBUG"));' 2>&1
	@$(PY) -c 'import kaggle_runner' || ( >&2 echo "kaggle_runner CANNOT be imported."; $(PY) -m pip install -e . && $(PY) -c 'import kaggle_runner')
	-@$(PY) -c 'from kaggle_runner.utils import AMQPURL, logger' 2>&1
	-@timeout 3s $(PY) -c 'import os; from kaggle_runner import logger; logger.debug("$@: DEBUG flag is %s", os.environ.get("DEBUG"));' 2>&1


.PHONY: mbd_log
mbd_log: ## mbd_log
	$(UNBUFFER) tail -f mbd_log | $(UNBUFFERP) xargs -ri -d '\n' -L 1 -I{} bash -c 'echo "$$(date): {}"'
.PHONY: mbd_interactive
mbd_interactive: multilang_bert_data.sh ## mbd_interactive XNLI data thing.
	bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log) &

.PHONY: dd
dd: kaggle ## Download datasets.
	@ eval "$$write_dataset_list_script"
	-mkdir -p /kaggle/input
	(cmp_name="jigsaw-multilingual-toxic-comment-classification"; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name; \
cd /kaggle/input/$$cmp_name; unzip '*.zip') &
	sed 's/"\(.*\)".*/\1/' .datasets | xargs -I{} bash -xc 'folder=$$(echo {} | sed "s/.*\///"); kaggle datasets download --unzip -p /kaggle/input/$${folder} {}' &

.PHONY: ddj
ddj: ## Download datasets of jigsaw.
	(cmp_name="jigsaw-unintended-bias-in-toxicity-classification"; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name; \
cd /kaggle/input/$$cmp_name; unzip '*.zip') &
	

.PHONY: vim
vim: ## vim
	-apt install vim -y && $(PY) -m pip install pyvim neovim jedi && yes | vim -u ~/.vimrc_back +PlugInstall +qa

.PHONY: kaggle
kaggle: /root/.kaggle/kaggle.json ## Copy Kaggle authentication.
	-@ xclip ~/.kaggle/kaggle.json -selection clipboard

.PHONY: update_sh_ipynb
update_sh_ipynb: ## update_sh_ipynb
	jupytext --sync hub/shonenkov_training_pipeline.ipynb || jupytext --set-formats ipynb,py hub/shonenkov_training_pipeline.ipynb

.PHONY: dmetadata
dmetadata: kaggle  ## dmetadata
	[ -d datas ] || (mkdir datas; kaggle datasets metadata -p datas/ k1gaggle/ml-bert-for-toxic-classfication-trained)

.PHONY: push_dataset
push_dataset: dmetadata ## Push dataset.
	-cp datas/dm.json datas/dm-metadata.json
	-ls *.bin | grep -v "last" | xargs -I{} mv {} datas/
	-cp node_submissions/* log.txt /kaggle/submission.csv datas
	kaggle datasets create -p datas/ #-m "$$(git show --no-patch --oneline) $$(date)"

.PHONY: /root/.kaggle/kaggle.json
/root/.kaggle/kaggle.json: ## /root/.kaggle/kaggle.json
	-@mkdir -p ~/.kaggle
	@grep "username" ~/.kaggle/kaggle.json || (printf "\e[?1004l"; echo "Please paste your kaggle API token"; cat > ~/.kaggle/kaggle.json </dev/tty)
	chmod 600 ~/.kaggle/kaggle.json

.PHONY: mbd_pretrain
mbd_pretrain: multilang_bert_data.sh apex ## mbd_pretrain
	-make tpu_setup
	STAGE=pretrain bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log

.PHONY: exit
exit: ## exit
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && make distclean
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && sleep 3 && (touch /tmp/rvs_exit && pkill ncat && pkill screen && pkill -f "rvs.sh") &

.PHONY: tpu_setup
tpu_setup: ## tpu_setup
	curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o /tmp/pytorch-xla-env-setup.py
	pip show torch_xla || $(PY) /tmp/pytorch-xla-env-setup.py #@param ["20200220","nightly", "xrt==1.15.0"]

.PHONY: xlmr
xlmr: ## xlmr
	$(PY) -m pip install --upgrade torch
	$(PY) -c "import torch; xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large');"

.PHONY: apex
apex: ## apex
	-$(PY) -m pip show apex || ([ -d /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a ] && \
$(PY) -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a)
	$(PY) -c "from apex import amp"

.PHONY: mbd
mbd: ## mbd
	$(UNBUFFER) make mbd_interactive >>mbd_log 2>&1 &
	make mbd_log

.PHONY: t
t: pccnct m ## Reverse Shell Server, mosh server.
	echo "Please check local mosh setup result"
	-$(IS_CENTOS) && sudo firewall-cmd --list-ports
	echo -e "\n\n\n\n\n\n\n\n\n"
	make push
	echo "Please check remote mosh setup result"
	-$(IS_CENTOS) && sudo firewall-cmd --list-ports

.PHONY: sshR
sshR: ## sshR
	if [ -d /content ]; then echo ssh -fNR 10010:$(KIP):9000 -p $(SSH_PORT) $(SERVER); \
else echo ssh -fNR 10010:$(KIP):8888 -p $(SSH_PORT) $(SERVER); fi
	-scp -P $(SSH_PORT) v@$(SERVER):~/.ssh/* ~/.ssh

.PHONY: sshRj
sshRj: ## sshRj
	$(PY) -m jupyter lab -h &>/dev/null || $(PY) -m pip install jupyterlab
	($(PY) -m jupyter lab --ip="$(KIP)" --port=9001 $(JUPYTER_PARAMS) || $(PY) -m jupyter lab --ip="$(KIP)" --port=9001 --allow-root) &
	ssh -fNR 10011:$(KIP):9001 -p $(SSH_PORT) $(SERVER)
	scp -P $(SSH_PORT) $(SERVER):~/.ssh/* ~/.ssh

.PHONY: githooks
githooks: ## githooks
	[ -f .git/hooks/pre-commit.sample ] && mv .git/hooks/pre-commit.sample .git/hooks/pre-commit && cat bin/pre-commit >> .git/hooks/pre-commit

.PHONY: distclean
distclean: clean ## distclean
	#-@git ls-files | sed 's/kaggle_runner\/\([^\/]*\)\/.*/\1/' | xargs -I{} sh -c "echo rm -rf {}; rm -rf {} 2>/dev/null"
	-@git ls-files | grep -v "\.md" | xargs -I{} sh -c 'echo rm "{}"; rm "{}"'
	-rm *.py *.sh *log
	-rm -r .git
	-rm -r __notebook_source__.ipynb bert gdrive_setup kaggle_runner.egg-info apex dotfiles rpt
	-find . -name "*.pyc" -print0 | xargs --null -I{} rm "{}"

.PHONY: ks
ks: ## List kernel sessions.
	curl -sSLG $(KIP):9000/api/sessions

.PHONY: Git push.
push_code: ## push_code
	-sed -i 's/https:\/\/\([^\/]*\)\//git@\1:/' .gitmodules
	-sed -i 's/https:\/\/\([^\/]*\)\//git@\1:/' .git/config
	git push

.PHONY: jigsaw-unintended-bias-in-toxicity-classification
jigsaw-unintended-bias-in-toxicity-classification: ## jigsaw-unintended-bias-in-toxicity-classification
	test ! -d /kaggle/input/jigsaw-unintended-bias-in-toxicity-classification

.PHONY: gpt2
gpt2: jigsaw-unintended-bias-in-toxicity-classification kaggle ## gpt2
	-mkdir -p /kaggle/input
	(cmp_name=$<; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name && \
cd /kaggle/input/$$cmp_name && unzip '*.zip') &

.PHONY: install_gitbook
install_gitbook: ## Setup gitbook.
	type gitbook &>/dev/null || ( \
curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh && \
sh nodesource_setup.sh && ( \
apt-get install -y nodejs; \
npm install -g gitbook-cli; \
npm install -g doctoc; \
npm install -g gitbook-summary; \
gitbook fetch 3.2.3 ; ) ) # fetch final stable version and add any requested plugins in book.json

.PHONY: setup_pip
setup_pip: ## setup_pip
	python3 -m pip -h &>/dev/null || (curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py)
	@poetry run python -m pip install --upgrade pip

.PHONY: setup_venv
setup_venv: setup_pip ## Install python3-venv
	-apt update && apt install -y python3-venv

.PHONY: pydoc
pydoc: setup_venv install_gitbook kr ## Set up pydoc and generate gitbook documentation.
	-apt install -y python3-pip
	$(PY) -m pip install pipx
	#pipx install 'pydoc-markdown>=3.0.0,<4.0.0'
	$(PY) -m pip install pydoc-markdown
	pipx install mkdocs
	#$$(head -n1 $$(which pydoc-markdown)  | sed 's/#!//') -m pip install -e .
	#$$(head -n1 ~/.local/bin/pydoc-markdown  | sed 's/#!//') -m pip install tensorflow
	$(PY) -m pip install -e .
	bash bin/document_thing
	-@rm kaggle_runner.md
	book sm -i node_modules
	sed -i 's/Your Book Title/Run your kernels/' SUMMARY.md
	@cat SUM*
	-gitbook install 
	-[ -f README.md ] || touch README.md
	-[ -f README ] || cp README.md README
	gitbook build . public # build to public path
	-tree public
	-timeout 360 gitbook serve public &
	#make distclean || true # no need, if we generate output to public folder

.PHONY: .PHONY
.PHONY: clean connect inner_lstm pc mbd_log

# Set your own project id here
# PROJECT_ID = 'your-google-cloud-project'
# from google.cloud import storage
# storage_client = storage.Client(project=PROJECT_ID)

.PHONY: sync_result
sync_result: ## sync_result
	while true; do git commit -asm "Good game" --no-edit; git pull; git push; sleep 10; done

.PHONY: d
d: ## Git diff.
	git diff; git diff --cached

.PHONY: install_template
install_template: ## install_template
	git submodule update --init
	git config --global init.templatedir '~/.git_template/template'
	"$$(git config --path --get init.templatedir)/../update.sh"
	"$$(git config --path --get init.templatedir)/configure.sh"

.PHONY: nodejs
nodejs: ## nodejs
	curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
	bash nodesource_setup.sh
	-apt-get install -y nodejs
	#apt install gcc g++ make


.PHONY: xla
xla: ## xla
	$(PY) -m pip show torch_xla || ( curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py; \
$(PY) pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev; \
$(PY) -m pip install *.whl; )

.PHONY: kr
kr: ## Download and install kaggle_runner.
	ssh-keyscan github.com >> githubKey
	ssh-keygen -lf githubKey
	mkdir -p ~/.ssh
	cat githubKey >> ~/.ssh/known_hosts
	rm githubKey
	[ -d kaggle_runner ] || (git clone https://github.com/pennz/kaggle_runner; \
mv kaggle_runner k && \
rsync -r k/* . ; rsync -r k/.* . ); \
git pull; \
git submodule update --init || ( \
sed -i 's/git@.*:/https:\/\/github.com\//' .git/config; \
sed -i 's/git@.*:/https:\/\/github.com\//' .gitmodules; \
git submodule update --init;); \
$(PY) -m pip show kaggle_runner || $(PY) -m pip install -e .;
	touch hub/custom_fastai_callbacks/__init__.py
	python3 -m pip show kaggle_runner || python3 -m pip install -e .;

.PHONY: entry
entry: kr ## entry
	export PATH=$$PWD/bin:$$PATH; pgrep -f entry || entry.sh &

.PHONY: prompt
prompt: ## prompt
	$(PY) -m pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall

.PHONY: sed
sed: ## sed
	@echo sed used is $(SED)

.PHONY: run_git_lab
run_git_lab: ## run_git_lab
	-pkill gitlab-runner
	/usr/lib/gitlab-runner/gitlab-runner run --working-directory /home/gitlab-runner \
--config /etc/gitlab-runner/config.toml --service gitlab-runner --syslog --user root &

.PHONY: gitlab
gitlab: ## gitlab
	curl -s https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
	apt install -y gitlab-runner
	( gitlab-runner register -n --executor shell -u \
https://gitlab.com/ -r _NCGztHrPW7T81Ysi_sS --name $$HOSTNAME --custom-run-args 'user = root'; \
sleep 5; \
make run_git_lab; \
while true; do pgrep 'gitlab-runner' || make run_git_lab; sleep 5; done & )

.PHONY: ide
ide: ## ide
	pipx install pydocstyle

.PHONY: help
help:  ## Print this help. ## help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

.PHONY: setup
setup:  setup_venv ## Setup the development environment for poetry (install dependencies).
	@if true; then \
		if ! which poetry &>/dev/null; then \
		  if ! which pipx &>/dev/null; then \
			  python3 -m pip install --user pipx; \
			fi; \
		  echo "Install poetry now"; \
		  python3 -m pipx install poetry; \
		  pipx ensurepath; \
		fi; \
	fi; \
	poetry install -v

.PHONY: check_all
check_all: check-docs check-code-quality check-types check-dependencies  ## Check it all!

.PHONY: check-code-quality
check-code-quality:  ## Check the code quality.
	@poetry run failprint -t "Checking code quality" -- flake8 --config=config/flake8.ini $(PY_SRC)

.PHONY: check-dependencies
check-dependencies:  ## Check for vulnerabilities in dependencies.
	@SAFETY=safety; \
	if ! $(CI); then \
		if ! which $$SAFETY &>/dev/null; then \
			SAFETY="pipx run safety"; \
		fi; \
	fi; \
	poetry export -f requirements.txt --without-hashes | \
		poetry run failprint --no-pty -t "Checking dependencies" -- $$SAFETY check --stdin --full-report

.PHONY: check-docs
check-docs:  ## Check if the documentation builds correctly.
	@poetry run failprint -t "Building documentation" -- mkdocs build -s

.PHONY: check-types
check-types:  ## Check that the code is correctly typed.
	@poetry run failprint -t "Type-checking" -- mypy --config-file config/mypy.ini $(PY_SRC)

.PHONY: changelog
changelog:  ## Update the changelog in-place with latest commits.
	@poetry run failprint -t "Updating changelog" -- python scripts/update_changelog.py \
		CHANGELOG.md "<!-- insertion marker -->" "^## \[(?P<version>[^\]]+)"

.PHONY: docs
docs: docs-regen kr ## Build the documentation locally.
	#$(PY) -m show mkdocs &>/dev/null || $(PY) -m pip install mkdocs mkdocs-material mkdocstrings
	python3 -m pip install mkdocs mkdocs-material mkdocstrings
	#@poetry run mkdocs build
	#$(PY) -m mkdocs build -d public
	python3 -m mkdocs build -d public

.PHONY: docs-py-md-gen
docs-py-md-gen:
	@poetry run bin/document_thing 1
	rm docs/kaggle_runner/runner_template/main.md
	rm docs/kaggle_runner/runners/tpu_trainer.md
	rm docs/kaggle_runner/datasets/mock_dataset.md

.PHONY: docs-regen
docs-regen: docs-py-md-gen setup_pip ## Regenerate some documentation pages.
	@poetry run python scripts/regen_docs.py

.PHONY: docs-serve
docs-serve: docs-regen  ## Serve the documentation (localhost:8000).
	@poetry run mkdocs serve

.PHONY: docs-deploy
docs-deploy: docs-regen  ## Deploy the documentation on GitHub pages.
	@poetry run mkdocs gh-deploy CHANGELOG.md "<!-- insertion marker -->" "^## \[(?P<version>[^\]]+)"

.PHONY: format
format:  ## Run formatting tools on the code.
	@poetry run failprint -t "Formatting code" -- black $(PY_SRC)
	@poetry run failprint -t "Ordering imports" -- isort -y -rc $(PY_SRC)

.PHONY: release
release:  ## Create a new release (commit, tag, push, build, publish, deploy docs).
ifndef v
	$(error Pass the new version with 'make release v=0.0.0')
endif
	@poetry run failprint -t "Bumping version" -- poetry version $(v)
	@poetry run failprint -t "Staging files" -- git add pyproject.toml CHANGELOG.md
	@poetry run failprint -t "Committing changes" -- git commit -m "chore: Prepare release $(v)"
	@poetry run failprint -t "Tagging commit" -- git tag v$(v)
	@poetry run failprint -t "Building dist/wheel" -- poetry build
	-@if ! $(CI) && ! $(TESTING); then \
		poetry run failprint -t "Pushing commits" -- git push; \
		poetry run failprint -t "Pushing tags" -- git push --tags; \
		poetry run failprint -t "Publishing version" -- poetry publish; \
		poetry run failprint -t "Deploying docs" -- poetry run mkdocs gh-deploy; \
	fi
