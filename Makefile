#export LD_LIBRARY_PATH := $(PWD)/lib:$(LD_LIBRARY_PATH)
export PATH := /nix/store/3ycgq0lva60yc2bw4qshmlsaqn0g90x4-nodejs-14.2.0/bin:$(HOME)/.local/bin:$(PWD)/bin:$(PATH)
export DEBUG := $(DEBUG)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43 

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

UNBUFFER := $(shell command -v unbuffer)
ifneq ($(UNBUFFER),)
	UNBUFFERP := $(UNBUFFER) -p
endif

KAGGLE_USER_NAME=$(shell jq -r '.username' ~/.kaggle/kaggle.json)
KIP=$(shell ip addr show dev eth0 | grep inet | sed 's/.*inet \([^\/]*\).*/\1/')


SED := $(shell which gsed &>/dev/null && echo "gsed")
ifeq ($(SED),)
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
PY=python3
SRC=$(wildcard */**.py)
SHELL=/bin/bash


IS_CENTOS=type firewall-cmd >/dev/null 2>&1

_: test
	@echo "DONE $@"

test: ctr
	@echo "DONE $@"

test_bert_torch: pytest
	if [ -z $$DEBUG ]; then $(PY) tests/test_bert_torch.py 2>&1 | $(UNBUFFERP) tee -a test_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT); \
else wt $(PY) -m pdb tests/test_bert_torch.py </dev/tty ; fi

pytest:
	$(PY) -m pip show pytest | grep "Version: 5." &>/dev/null || ($(PY) -m pip install --upgrade pytest && $(PY) -m pip install --upgrade pytest-cov)

check_log_receiver:
	@echo "$@" will use tcp to receive logs
	-pkill -f "$(CHECK_PORT)"
	-$(IS_CENTOS) && (pgrep -f firewalld >/dev/null || sudo systemctl start firewalld)
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp --permanent
	ncat -vkl --recv-only  -p $(CHECK_PORT) -o logs_check & sleep 1; tail -f logs_check # logs_check will be used by pcc to get mosh-client connect authentication info

pc:
	pcc
	make connect

m:
	while true; do (setup_mosh_server 2>&1 | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT)) & sleep $$((60*25)); done

mosh:
	( while true; do bash -x setup_mosh_server& [ -f /tmp/mexit ] && exit 0; sleep 600; done 2>&1 | $(UNBUFFERP) tee -a ms_connect_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT) ) &
	#@sleep 1
	#tail ms_connect_log

rvs_session:
	-tmux new-session -d -n "good-day" -s rvsConnector "cat"
	-tmux set-option -t rvsConnector renumber-windows on

_pccnct:
	-pkill -f "50001.*addNew"
	echo "start mosh connector";
	$(UNBUFFER) ncat -uklp 50001 -c "bash -c 'echo $$(date): New Incoming >>mosh_log'; echo; addNewNode.sh mosh" &
	echo "connection listener setup done."
	echo "pccnct has been put to backgound."
	
pccnct: rvs_session _pccnct
	make check_log_receiver & # will output to current process
	-$(IS_CENTOS) && sudo service rabbitmq-server start # For AMQP log, our server 
	@echo "pc connector started now"

ctr: kr check install_dep pytest $(SRC)
	-timeout 10 git push
	[ -f bin/cc-test-reporter ] || curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > bin/cc-test-reporter
	chmod +x bin/cc-test-reporter
	-bin/cc-test-reporter before-build
	-$(PY) -m coverage run -m pytest -vs --full-trace tests
	-$(PY) -m coverage report -m -i | grep '^TOTAL.*[0-9]\{1,\}'
	-$(PY) -m coverage xml -i -o coverage.xml
	-bin/cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT

get_submission:
	kaggle datasets download --file submission.csv --unzip k1gaggle/bert-for-toxic-classfication-trained
	-unzip '*.zip' && rm *.zip && mv *.csv submission.csv

push: rvs_session $(SRC)
	-#git push # push first as kernel will download the codes, so put new code to github first
	-@echo "$$(which $(PY)) is our $(PY) executable"; [[ x$$(which $(PY)) =~ conda ]]
	sed -i 's/\(id": "\)\(.*\)\//\1$(KAGGLE_USER_NAME)\//' kaggle_runner/runner_template/kernel-metadata.json
	title=$$(git show --no-patch --oneline | tr " " "_"); sed -i 's/title\(.*\)|.*"/title\1| '$$title\"/ kaggle_runner/runner_template/kernel-metadata.json
	git add kaggle_runner/runner_template/kernel-metadata.json && git commit -sm "Update metadata when push to server" --no-gpg && git push &
	run_coordinator $(PHASE) # source only works in specific shell: bash or ...

connect:
	tmux select-window -t rvsConnector:{end}
	tmux switch -t rvsConnector:{end}


lint: $(SRC)
	echo $(SRC)
	pylint -E $(SRC)

inner_lstm:
	while true; do test x$$(git pull | grep -c Already) = x1 || { $(PY) \
lstm.py 2>&1 | tee -a lstm_log; };  echo "$$(date) $$HOSTNAME CPU: "$$(grep \
'cpu ' /proc/stat >/dev/null;sleep 0.1; grep 'cpu ' /proc/stat | awk -v RS='' \
'{print ($$13-$$2+$$15-$$4)*100/($$13-$$2+$$15-$$4+$$16-$$5)}')% 'Mem: '$$(awk \
'/MemTotal/{t=$$2}/MemAvailable/{a=$$2}END{print 100-100*a/t}' /proc/meminfo)% \
'Uptime: '$$(uptime | awk '{print $$3}'); sleep 10; done

lstm:
	-pkill -f "inner_lstm"
	make inner_lstm

debug_toxic:
	DEBUG=true make toxic #python3 -m pdb $$(which pytest) -sv tests/test_distilbert_model.py

toxic: check install_dep
	echo $$(ps aux | grep "make $@$$")
	echo "$@:" DEBUG flag is $$DEBUG .
	bash -c 'ppid=$$PPID; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); while [[ -n "$$mpid" ]] && [[ "$$mpid" -lt "$$((ppid-10))" ]]; do if [ ! -z $$mpid ]; then echo "we will kill existing \"make $@\" with pid $$mpid"; kill -9 $$mpid; sleep 1; else return 0; fi; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); done'
	if [ -z $$DEBUG ]; then $(UNBUFFER) $(PY) tests/test_distilbert_model.py 2>&1 | $(UNBUFFERP) tee -a toxic_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT); else wt '$(PY) -m ipdb tests/test_distilbert_model.py'; fi
	-git stash pop || true

test_coor: update_code $(SRC)
	$(PY) -m pytest -k "test_generate_runner" tests/test_coord.py; cd .runners/intercept-resnet-384/ && $(PY) main.py

clean:
	#-bash -c 'currentPpid=$$(pstree -spa $$$$ | $(SED) -n "2,3 p" |  cut -d"," -f 2 | cut -d" " -f 1); pgrep -f "rvs.sh" | sort | grep -v -e $$(echo $$currentPpid | $(SED) "s/\s\{1,\}/ -e /" ) -e $$$$ | xargs -I{} kill -9 {}'
	-ps aux | grep "ncat .*lp" | grep -v "while" | grep -v "50001" | grep -v "grep" | tee /dev/tty | awk '{print $$2} ' | xargs -I{} kill -9 {}
	-rm -rf __pycache__ mylogs dist/* build/*


submit:
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation
run_submit:
	$(PY) DAF3D/Train.py
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

twine:
	@$(PY) -m twine -h >/dev/null || ( echo "twine not found, will install it." ; $(PY) -m pip install --user --upgrade twine )
publish: clean twine
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
update_code:
	#-git stash;
	git pull
install_dep_seg:
	bash -c '(test -z "$$($(PY) -m albumentations 2>&1 | grep direct)" && $(PY) -m pip install -U git+https://github.com/albu/albumentations) & \
(test -z "$$($(PY) -m segmentation_models_pytorch 2>&1 | grep direct)" && $(PY) -m pip install git+https://github.com/qubvel/segmentation_models.pytorch) & \
wait'


$(TOXIC_DEP):
	@echo "Installing $@"
	#$(PY) -m pip show $@ &>/dev/null || $(PY) -m pip install -q $@

install_dep: $(TOXIC_DEP) pytest
	for p in $^; do ($(PY) -m pip show $$p &>/dev/null || $(PY) -m pip install -q $$p) & done; wait
	#$(PY) -m pip install -q eumetsat expect &
	#conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago
	@echo make $@ $^ done

connect_close:
	stty raw -echo && ( ps aux | $(SED) -n 's/.*vvlp \([0-9]\{1,\}\)/\1/p' | xargs -I{} ncat 127.1 {} )

rpdbrvs:
	while true; do ncat $(SERVER) 23454 --sh-exec 'ncat -w 3 127.1 4444; echo \# nc return $?' ; sleep 1; echo -n "." ; done;

rvs:
	while true; do ncat -w 60s -i 1800s $(SERVER) $$(( $(CHECK_PORT) - 1 )) -c "echo $$(date) \
started connection; echo $$HOSTNAME; python -c 'import pty; \
pty.spawn([\"/bin/bash\", \"-li\"])'"; echo "End of one rvs."; sleep 5; done;

r:
	bash -c "SAVED_STTY=$$(stty -g); stty raw -echo; ncat -v $(SERVER) $$(( $(CHECK_PORT) - 1 )); stty $$SAVED_STTY"

dbroker:
	stty raw && while true; do echo "Start Listening"; ncat --broker -v -m 2 -p $$(( $(CHECK_PORT) - 1 )); echo >&2 "Listen failed, will restart again." ; sleep 5; done  # just one debug session at a time, more will make you confused

rpdbc:
	bash -c "SAVED_STTY=$$(stty -g); stty onlcr onlret -icanon opost -echo -echoe -echok -echoctl -echoke; ncat -v 127.0.0.1 23454; stty $$SAVED_STTY"

mq:
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

amqp_log:
	-$(IS_CENTOS) && sudo systemctl restart rabbitmq-server.service
	$(UNBUFFER) receive_logs_topic \*.\* 2>&1 | $(UNBUFFERP) tee -a mq_log | $(UNBUFFERP) $(SED) -n 's/^.*\[x\] \(.*\)/\1/p'  | (type jq >/dev/null 2>&1 && $(UNBUFFERP) jq -r '.msg' || $(UNBUFFERP) cat -)
	# sleep 3; tail -f mq_log | $(SED) -n "s/\(.*\)\[x.*/\1/p"

mlocal:
	tty_config=$$(stty -g); size=$$(stty size); $(MC); stty $$tty_config; stty columns $$(echo $$size | cut -d" " -f 2) rows $$(echo $$size | cut -d" " -f 1)

check:
	-ps aux | grep make
	echo sed $(SED)
	echo PATH $(PATH)
	-@echo $(http_proxy)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT) | ncat $(SERVER) $(CHECK_PORT)
	-expect -h
	pstree -laps $$$$
	-@echo "$$(which $(PY)) is our $(PY) executable"; if [[ x$$(which $(PY)) =~ conda ]]; then echo conda env fine; else echo >&2 conda env not set correctly, please check.; source ~/.bashrc; conda activate pyt; fi
	@$(PY) -c 'import os; print("$@: DEBUG=%s" % os.environ.get("DEBUG"));' 2>&1
	@$(PY) -c 'import kaggle_runner' || ( >&2 echo "kaggle_runner CANNOT be imported."; $(PY) -m pip install -e . && $(PY) -c 'import kaggle_runner')
	-@$(PY) -c 'from kaggle_runner.utils import AMQPURL, logger' 2>&1
	-@timeout 3s $(PY) -c 'import os; from kaggle_runner import logger; logger.debug("$@: DEBUG flag is %s", os.environ.get("DEBUG"));' 2>&1


mbd_log:
	$(UNBUFFER) tail -f mbd_log | $(UNBUFFERP) xargs -ri -d '\n' -L 1 -I{} bash -c 'echo "$$(date): {}"'
mbd_interactive: multilang_bert_data.sh
	bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log) &

dd: kaggle
	@ eval "$$write_dataset_list_script"
	-mkdir -p /kaggle/input
	(cmp_name="jigsaw-multilingual-toxic-comment-classification"; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name; \
cd /kaggle/input/$$cmp_name; unzip '*.zip') &
	sed 's/"\(.*\)".*/\1/' .datasets | xargs -I{} bash -xc 'folder=$$(echo {} | sed "s/.*\///"); kaggle datasets download --unzip -p /kaggle/input/$${folder} {}' &

ddj:
	(cmp_name="jigsaw-unintended-bias-in-toxicity-classification"; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name; \
cd /kaggle/input/$$cmp_name; unzip '*.zip') &
	

vim:
	-apt install vim -y && $(PY) -m pip install pyvim neovim jedi && yes | vim -u ~/.vimrc_back +PlugInstall +qa

kaggle: /root/.kaggle/kaggle.json
	-@ xclip ~/.kaggle/kaggle.json -selection clipboard

update_sh_ipynb:
	jupytext --sync hub/shonenkov_training_pipeline.ipynb || jupytext --set-formats ipynb,py hub/shonenkov_training_pipeline.ipynb

dmetadata: kaggle 
	[ -d datas ] || (mkdir datas; kaggle datasets metadata -p datas/ k1gaggle/ml-bert-for-toxic-classfication-trained)

push_dataset: dmetadata
	-cp datas/dm.json datas/dm-metadata.json
	-ls *.bin | grep -v "last" | xargs -I{} mv {} datas/
	-cp node_submissions/* log.txt /kaggle/submission.csv datas
	kaggle datasets create -p datas/ #-m "$$(git show --no-patch --oneline) $$(date)"

/root/.kaggle/kaggle.json:
	-@mkdir -p ~/.kaggle
	@grep "username" ~/.kaggle/kaggle.json || (printf "\e[?1004l"; echo "Please paste your kaggle API token"; cat > ~/.kaggle/kaggle.json </dev/tty)
	chmod 600 ~/.kaggle/kaggle.json

mbd_pretrain: multilang_bert_data.sh apex
	-make tpu_setup
	STAGE=pretrain bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log

exit:
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && make distclean
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && sleep 3 && (touch /tmp/rvs_exit && pkill ncat && pkill screen && pkill -f "rvs.sh") &

tpu_setup:
	curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o /tmp/pytorch-xla-env-setup.py
	pip show torch_xla || $(PY) /tmp/pytorch-xla-env-setup.py #@param ["20200220","nightly", "xrt==1.15.0"]

xlmr:
	$(PY) -m pip install --upgrade torch
	$(PY) -c "import torch; xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large');"

apex:
	-$(PY) -m pip show apex || ([ -d /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a ] && \
$(PY) -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a)
	$(PY) -c "from apex import amp"

mbd:
	$(UNBUFFER) make mbd_interactive >>mbd_log 2>&1 &
	make mbd_log

dataset: mbd
	-mkdir .k && mv * .* .k && mv .k/toxic*pkl . && rm -r .k

p:
	pushd kaggle_runner/hub/bert && (git commit -asm "GG" --no-gpg || true) && git push && popd && git add kaggle_runner/hub/bert && git commit -sm "Updated bert" --no-gpg && git push
pl:
	git stash; git pull; git submodule update --init

t: pccnct m
	echo "Please check local mosh setup result"
	-$(IS_CENTOS) && sudo firewall-cmd --list-ports
	echo -e "\n\n\n\n\n\n\n\n\n"
	make push
	echo "Please check remote mosh setup result"
	-$(IS_CENTOS) && sudo firewall-cmd --list-ports

sshR:
	if [ -d /content ]; then echo ssh -fNR 10010:$(KIP):9000 -p $(SSH_PORT) $(SERVER); \
else echo ssh -fNR 10010:$(KIP):8888 -p $(SSH_PORT) $(SERVER); fi
	-scp -P $(SSH_PORT) v@$(SERVER):~/.ssh/* ~/.ssh

sshRj:
	$(PY) -m jupyter lab -h &>/dev/null || $(PY) -m pip install jupyterlab
	($(PY) -m jupyter lab --ip="$(KIP)" --port=9001 $(JUPYTER_PARAMS) || $(PY) -m jupyter lab --ip="$(KIP)" --port=9001 --allow-root) &
	ssh -fNR 10011:$(KIP):9001 -p $(SSH_PORT) $(SERVER)
	scp -P $(SSH_PORT) $(SERVER):~/.ssh/* ~/.ssh

githooks:
	[ -f .git/hooks/pre-commit.sample ] && mv .git/hooks/pre-commit.sample .git/hooks/pre-commit && cat bin/pre-commit >> .git/hooks/pre-commit

distclean: clean
	#-@git ls-files | sed 's/kaggle_runner\/\([^\/]*\)\/.*/\1/' | xargs -I{} sh -c "echo rm -rf {}; rm -rf {} 2>/dev/null"
	-@git ls-files | grep -v "\.md" | xargs -I{} sh -c 'echo rm "{}"; rm "{}"'
	-rm *.py *.sh *log
	-rm -r .git
	-rm -r __notebook_source__.ipynb bert gdrive_setup kaggle_runner.egg-info apex dotfiles rpt
	-find . -name "*.pyc" -print0 | xargs --null -I{} rm "{}"

ks:
	curl -sSLG $(KIP):9000/api/sessions

push_code:
	-sed -i 's/https:\/\/\([^\/]*\)\//git@\1:/' .gitmodules
	-sed -i 's/https:\/\/\([^\/]*\)\//git@\1:/' .git/config
	git push

jigsaw-unintended-bias-in-toxicity-classification:
	test ! -d /kaggle/input/jigsaw-unintended-bias-in-toxicity-classification

gpt2: jigsaw-unintended-bias-in-toxicity-classification kaggle
	-mkdir -p /kaggle/input
	(cmp_name=$<; \
kaggle competitions download -p /kaggle/input/$$cmp_name $$cmp_name && \
cd /kaggle/input/$$cmp_name && unzip '*.zip') &

install_gitbook:
	type gitbook &>/dev/null || ( \
curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh && \
sh nodesource_setup.sh && ( \
apt-get install -y nodejs; \
npm install -g gitbook-cli; \
npm install -g doctoc; \
npm install -g gitbook-summary; \
gitbook fetch 3.2.3 ; ) ) # fetch final stable version and add any requested plugins in book.json

setup_pip:
	$(PY) -m pip || apt install -y python3-pip

pydoc: setup_pip install_gitbook kr
	-apt install -y python3-pip
	$(PY) -m pip install pipx
	-apt-get install -y python3-venv || yum install -y python3-venv
	#pipx install 'pydoc-markdown>=3.0.0,<4.0.0'
	$(PY) -m pip install pydoc-markdown
	pipx ensurepath
	pipx install mkdocs
	$$(head -n1 $$(which pydoc-markdown)  | sed 's/#!//') -m pip install -e .
	#$$(head -n1 ~/.local/bin/pydoc-markdown  | sed 's/#!//') -m pip install tensorflow
	bash bin/document_thing
	-@rm kaggle_runner.md
	book sm -i node_modules
	sed -i 's/Your Book Title/Run your kernels/' SUMMARY.md
	@cat SUM*
	-gitbook install 
	-[ -f README.md ] || touch README.md
	-gitbook build . public # build to public path
	-timeout 360 gitbook serve public &
	-make distclean || true

.PHONY: clean connect inner_lstm pc mbd_log

# Set your own project id here
# PROJECT_ID = 'your-google-cloud-project'
# from google.cloud import storage
# storage_client = storage.Client(project=PROJECT_ID)

sync_result:
	while true; do git commit -asm "Good game" --no-edit; git pull; git push; sleep 10; done

d:
	git diff; git diff --cached

install_template:
	git submodule update --init
	git config --global init.templatedir '~/.git_template/template'
	"$$(git config --path --get init.templatedir)/../update.sh"
	"$$(git config --path --get init.templatedir)/configure.sh"

nodejs:
	curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
	bash nodesource_setup.sh
	-apt-get install -y nodejs
	#apt install gcc g++ make


xla:
	$(PY) -m pip show torch_xla || ( curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py; \
$(PY) pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev; \
$(PY) -m pip install *.whl; )

kr:
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

entry: kr
	export PATH=$$PWD/bin:$$PATH; pgrep -f entry || entry.sh &

prompt:
	$(PY) -m pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall

sed:
	@echo sed used is $(SED)

run_git_lab:
	-pkill gitlab-runner
	/usr/lib/gitlab-runner/gitlab-runner run --working-directory /home/gitlab-runner \
--config /etc/gitlab-runner/config.toml --service gitlab-runner --syslog --user root

gitlab:
	curl -s https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
	apt install -y gitlab-runner
	pgrep gitlab-runner &>/dev/null || ( gitlab-runner register -n --executor shell -u \
https://gitlab.com/ -r _NCGztHrPW7T81Ysi_sS --name $$HOSTNAME --custom-run-args 'user = root'; \
sleep 5; \
make run_git_lab; \
while true; do pgrep 'gitlab-runner' || make run_git_lab; sleep 5; done & )

ide:
	pipx install pydocstyle
