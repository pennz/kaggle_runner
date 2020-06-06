#export LD_LIBRARY_PATH := $(PWD)/lib:$(LD_LIBRARY_PATH)
export PATH := $(PWD)/reversShells:$(PATH)
export DEBUG := $(DEBUG)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43 

UNBUFFER := $(shell command -v unbuffer)
ifneq ($(UNBUFFER),)
	UNBUFFERP := $(UNBUFFER) -p
endif

KAGGLE_USER_NAME=$(shell jq -r '.username' ~/.kaggle/kaggle.json)

SED := $(shell type -p gsed)
ifeq ($(SED),)
	SED := $(shell tpye -p sed)
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
export CHECK_PORT

URL="https://www.kaggleusercontent.com/kf/33961266/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..b3ZzhVJx_c1vhjL3vVc5Ow.4i-Vpk1-bF9zCZJP7LHiuSY44ljoCyKbD7rLcvDSUuViAHL3Xw_Idb3gkMIGhqY6kLN9GX2VzGdxAv9qqOJGXYc7EUeljbX6dvjdssk5Iuhwl4kxz-TIsWYaxqONbMGBQX9rT-nIJYmpjV8UKle7DlX1UYFJKhLYyuckV1B5ZEGHkRjdzwasPlhc8IJkX83RfLhe7C6T0pR8oFU-gmvtQxSvKzXprbYvPQVRMyBf4xD8Bm9xvEq8aFVIiwHGROwvIcorUhZ3cHsCXRSE6RDm7f1rmbA_52xetuCEB2de1_tg-XZ7FoBx6_QaQHXnZWWRhZ1Edyzt5LlakbQI55Ncq3RBByr84QnJmAc9yJORqorQrtEWuAXCrHbYTiKR39i4sm2mkcvIhdgqYuHh8E7ZMXt7MiYr4W6Na233NBRPzY4l15DXqV5ZXp_m-th1ljwxUK8AvNTo0Qs3PNd0bvezFQew10jrMR-N-Z8ZFqtX--Ba8BbMFex6_jJxhN6JXFOXPwCJUWhrZ1yYNE3iqpavJkOM06Vkx6UEOhNbawmPrDtzF4vXViCdHbfUTcpd2qvmXgVlTg7cULSw4MzGdN-Uqbp6-MnpvGIFrRVOVooRE5u8zhrbRcZL4RApjr9SrIEPm1WSp7Qlj8wjktBL4K1bNKn4NE9-AFtOu_0X-lL0Afav41RxxhqQyL_Ox3o3YI8Y.hz022ycDLUciahf-YOeEDw/inceptionresnetv2-520b38e4.pth"
PY=python3
SRC=$(wildcard */**.py)
SHELL=/bin/bash

RUN_PC=cnt=$$(pgrep -f "50001.*addNew" | wc -l); echo $$cnt; [ $$cnt -lt 3 ] && \
( echo "start mosh connector"; \
$(UNBUFFER) ncat -uklp 50001 -c "echo $$(date): New Incoming >>mosh_log; bash -x reversShells/addNewNode.sh mosh"; \
echo "connection done." )

IS_CENTOS=type firewall-cmd >/dev/null 2>&1

_: test_bert_torch
	echo "DONE"
	#kill 7 8 # magic pids

test: test_bert_torch
	echo "TEST DONE"

test_bert_torch: pytest
	$(PY) -m pytest -s -k "Test_bert_multi_lang" tests/test_bert_torch.py | $(UNBUFFERP) tee -a test_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT)

pytest:
	$(PY) -m pip show pytest | grep "Version: 5.0" &>/dev/null || $(PY) -m pip install pytest==5.0

log_receiver:
	@echo "$@" will use tcp to receive logs
	-pkill -f "$(CHECK_PORT)"
	-$(IS_CENTOS) && (pgrep -f firewalld >/dev/null || sudo systemctl start firewalld)
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp
	-$(IS_CENTOS) && sudo firewall-cmd --add-port $(CHECK_PORT)/tcp --permanent
	ncat -vkl --recv-only  -p $(CHECK_PORT) -o logs_check & sleep 1; tail -f logs_check # logs_check will be used by pcc to get mosh-client connect authentication info

pc:
	./pcc
	make connect

mosh:
	while true; do (./setup_mosh_server 2>&1 | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT)) & sleep $$((60*25)); done

m:
	( while true; do bash -x ./setup_mosh_server& [ -f /tmp/mexit ] && exit 0; sleep 600; done 2>&1 | $(UNBUFFERP) tee -a ms_connect_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT) ) &
	#@sleep 1
	#tail ms_connect_log

rvs_session:
	-tmux new-session -d -n "good-day" -s rvsConnector "cat"
	-tmux set-option -t rvsConnector renumber-windows on

_pccnct:
	bash -xc '$(RUN_PC)' &  # for mosh, start listen instances, use 50001/udp and 9xxx/udp
	echo "pccnct has been put to backgound"
	
pccnct: rvs_session _pccnct
	make log_receiver & # will output to current process
	-$(IS_CENTOS) && sudo service rabbitmq-server start # For AMQP log, our server 
	@echo "pc connector started now"

all: $(SRC)
	-git push
	[ -f ./cc-test-reporter ] || curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > ./cc-test-reporter
	chmod +x ./cc-test-reporter
	./cc-test-reporter before-build
	-coverage run -m pytest .
	coverage xml
	./cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT

push: rvs_session $(SRC)
	-#git push # push first as kernel will download the codes, so put new code to github first
	-@echo "$$(which $(PY)) is our $(PY) executable"; [[ x$$(which $(PY)) =~ conda ]]
	sed -i 's/\(id": "\)\(.*\)\//\1$(KAGGLE_USER_NAME)\//' kaggle_runner/runner_template/kernel-metadata.json
	title=$$(git show --no-patch --oneline | tr " " "_"); sed -i 's/title\(.*\)|.*"/title\1| '$$title\"/ kaggle_runner/runner_template/kernel-metadata.json
	git add kaggle_runner/runner_template/kernel-metadata.json && git commit -sm "Update metadata when push to server" --no-gpg && git push &
	./run_coordinator $(PHASE) # source only works in specific shell: bash or ...

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

wt:
	chmod +x wt

toxic: wt check
	echo $$(ps aux | grep "make $@$$")
	echo DEBUG flag is $$DEBUG .
	bash -c 'ppid=$$PPID; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); while [[ -n "$$mpid" ]] && [[ "$$mpid" -lt "$$((ppid-10))" ]]; do if [ ! -z $$mpid ]; then echo "we will kill existing \"make $@\" with pid $$mpid"; kill -9 $$mpid; sleep 1; else return 0; fi; mpid=$$(pgrep -f "make $@$$" | sort | head -n 1); done'
	if [ -z $$DEBUG ]; then $(UNBUFFER) $(PY) tests/test_distilbert_model.py 2>&1 | $(UNBUFFERP) tee -a toxic_log | $(UNBUFFERP) ncat --send-only $(SERVER) $(CHECK_PORT); else ./wt '$(PY) -m ipdb tests/test_distilbert_model.py'; fi
	-git stash pop || true

test: update_code $(SRC)
	eval 'echo $$(which $(PY)) is our $(PY) executable'
	$(PY) -m pytest -k "test_generate_runner" tests/test_coord.py; cd .runners/intercept-resnet-384/ && $(PY) main.py
clean:
	#-bash -c 'currentPpid=$$(pstree -spa $$$$ | $(SED) -n "2,3 p" |  cut -d"," -f 2 | cut -d" " -f 1); pgrep -f "rvs.sh" | sort | grep -v -e $$(echo $$currentPpid | $(SED) "s/\s\{1,\}/ -e /" ) -e $$$$ | xargs -I{} kill -9 {}'
	-ps aux | grep "vlp" | grep -v "while" | grep -v "grep" | tee /dev/tty | awk '{print $$2} ' | xargs -I{} kill -9 {}
	-rm -rf __pycache__ mylogs dist/* build/*


submit:
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation
run_submit:
	$(PY) DAF3D/Train.py
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

twine:
	@python3 -m twine -h >/dev/null || ( echo "twine not found, will install it." ; python3 -m pip install --user --upgrade twine )
publish: clean twine
	if [[ x$(TAG) =~ xv ]] || [ -z $(TAG) ]; then >&2 echo "Please pass TAG \
flag when you call make, and use something like 0.0.3, not v0.0.3"; false; else \
gsed -i 's/version=.*/version=\"$(TAG)\",/' setup.py || \
$(SED) -i 's/version=.*/version=\"$(TAG)\",/' setup.py ;\
git add setup.py && \
(git tag -d "v$(TAG)"; git push --delete origin "v$(TAG)" || true) && \
git commit -sm "setup.py: v$(TAG)" && git tag -s "v$(TAG)" && git push --tags; \
fi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
update_code:
	#-git stash;
	git pull
install_dep_seg:
	bash -c '(test -z "$$($(PY) -m albumentations 2>&1 | grep direct)" && $(PY) -m pip install -U git+https://github.com/albu/albumentations) & \
(test -z "$$($(PY) -m segmentation_models_pytorch 2>&1 | grep direct)" && $(PY) -m pip install git+https://github.com/qubvel/segmentation_models.pytorch) & \
wait'

install_dev_dep:
	$(PY) -m pip install kaggle

install_dep:
	bash -c '$(PY) -m pip install -q ipdb & \
$(PY) -m pip install -q pyicu & \
$(PY) -m pip install -q pycld2 & \
$(PY) -m pip install -q polyglot & \
$(PY) -m pip install -q textstat & \
$(PY) -m pip install -q googletrans & \
wait'
	#$(PY) -m pip install -q eumetsat expect &
	#conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago

connect_close:
	stty raw -echo && ( ps aux | $(SED) -n 's/.*vvlp \([0-9]\{1,\}\)/\1/p' | xargs -I{} ncat 127.1 {} )

rpdbrvs:
	while true; do ncat $(SERVER) 23454 --sh-exec 'ncat -w 3 127.1 4444; echo \# nc return $?' ; sleep 1; echo -n "." ; done;
rpdbs:
	while true; do ncat -vlp 23454; sleep 1; done  # just one debug session at a time, more will make you confused
rpdbc:
	bash -c "SAVED_STTY=$$(stty -g); stty onlcr onlret -icanon opost -echo -echoe -echok -echoctl -echoke; ncat -v 127.0.0.1 23454; stty $$SAVED_STTY"

mq:
	make amqp_log &
	id -u rabbitmq &>/dev/null && while [ $$(ps -u rabbitmq | wc -l) -lt 5 ]; do sleep 60; ps aux | grep "amqp" | tee /dev/tty |  grep -v -e "sh" -e "grep" | awk '{print $$2} ' | xargs -I{} kill {}; make amqp_log &; jobs; done

amqp_log:
	-$(IS_CENTOS) && sudo systemctl restart rabbitmq-server.service
	$(UNBUFFER) ./receive_logs_topic \*.\* 2>&1 | $(UNBUFFERP) tee -a mq_log | $(UNBUFFERP) $(SED) -n 's/^.*\[x\] \(.*\)/\1/p'  | (type jq >/dev/null 2>&1 && $(UNBUFFERP) jq -r '.msg' || $(UNBUFFERP) cat -)
	# sleep 3; tail -f mq_log | $(SED) -n "s/\(.*\)\[x.*/\1/p"

mlocal:
	tty_config=$$(stty -g); size=$$(stty size); $(MC); stty $$tty_config; stty columns $$(echo $$size | cut -d" " -f 2) rows $$(echo $$size | cut -d" " -f 1)

check:
	-ps aux | grep make
	-@echo $(http_proxy)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT)
	-@echo $(UNBUFFER) $(UNBUFFERP) $(SERVER) $(CHECK_PORT) | ncat $(SERVER) $(CHECK_PORT)
	-expect -h
	pstree -laps $$$$
	-@echo "$$(which $(PY)) is our $(PY) executable"; if [[ x$$(which $(PY)) =~ conda ]]; then echo conda env fine; else echo >&2 conda env not set correctly, please check.; source ~/.bashrc; conda activate pyt; fi
	@$(PY) -c 'import os; print("DEBUG=%s" % os.environ.get("DEBUG"));' 2>&1
	@$(PY) -c 'import kaggle_runner' || ( >&2 echo "kaggle_runner cannot be imported."; $(PY) -m pip install -e . && $(PY) -c 'import kaggle_runner')
	@$(PY) -c 'from kaggle_runner.utils import AMQPURL, logger' 2>&1
	-@timeout 3s $(PY) -c 'import os; from kaggle_runner import logger; logger.debug("DEBUG flag is %s", os.environ.get("DEBUG"));' 2>&1


mbd_log:
	$(UNBUFFER) tail -f mbd_log | $(UNBUFFERP) xargs -ri -d '\n' -L 1 -I{} bash -c 'echo "$$(date): {}"'
mbd_interactive: multilang_bert_data.sh
	bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log) &

mbd_pretrain: multilang_bert_data.sh may_torch_gpu_setup 
	-make tpu_setup
	STAGE=pretrain bash -x multilang_bert_data.sh 2>&1 | tee -a mbd_i_log

exit:
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && make distclean
	@[ -z "$${DEBUG}" ] && type nvidia-smi &>/dev/null && sleep 3 && (touch /tmp/rvs_exit && pkill ncat && pkill -f "rvs.sh") &

tpu_setup:
	curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o /tmp/pytorch-xla-env-setup.py
	pip show torch_xla || $(PY) /tmp/pytorch-xla-env-setup.py #@param ["20200220","nightly", "xrt==1.15.0"]

may_torch_gpu_setup:
	-$(PY) -m pip show apex || ([ -d /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a ] && \
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a)
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

githooks:
	[ -f .git/hooks/pre-commit.sample ] && mv .git/hooks/pre-commit.sample .git/hooks/pre-commit && cat bin/pre-commit >> .git/hooks/pre-commit

distclean: clean
	-@git ls-files | sed 's/kaggle_runner\/\([^\/]*\)\/.*/\1/' | xargs -I{} sh -c "echo rm -rf {}; rm -rf {} 2>/dev/null"
	-@git ls-files | xargs -I{} sh -c 'echo rm -r $$(dirname {}); rm -r $$(dirname {}) 2>/dev/null'
	rm *.py *.sh *log
	rm -r .git
	rm -r __notebook_source__.ipynb bert gdrive_setup kaggle_runner.egg-info apex dotfiles  kaggle_runner rpt

dataset:
	kaggle datasets download -d k1gaggle/toxic-multilang-trained-torch-model

.PHONY: clean connect inner_lstm pc mbd_log
