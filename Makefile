export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43

URL="https://www.kaggleusercontent.com/kf/33961266/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..b3ZzhVJx_c1vhjL3vVc5Ow.4i-Vpk1-bF9zCZJP7LHiuSY44ljoCyKbD7rLcvDSUuViAHL3Xw_Idb3gkMIGhqY6kLN9GX2VzGdxAv9qqOJGXYc7EUeljbX6dvjdssk5Iuhwl4kxz-TIsWYaxqONbMGBQX9rT-nIJYmpjV8UKle7DlX1UYFJKhLYyuckV1B5ZEGHkRjdzwasPlhc8IJkX83RfLhe7C6T0pR8oFU-gmvtQxSvKzXprbYvPQVRMyBf4xD8Bm9xvEq8aFVIiwHGROwvIcorUhZ3cHsCXRSE6RDm7f1rmbA_52xetuCEB2de1_tg-XZ7FoBx6_QaQHXnZWWRhZ1Edyzt5LlakbQI55Ncq3RBByr84QnJmAc9yJORqorQrtEWuAXCrHbYTiKR39i4sm2mkcvIhdgqYuHh8E7ZMXt7MiYr4W6Na233NBRPzY4l15DXqV5ZXp_m-th1ljwxUK8AvNTo0Qs3PNd0bvezFQew10jrMR-N-Z8ZFqtX--Ba8BbMFex6_jJxhN6JXFOXPwCJUWhrZ1yYNE3iqpavJkOM06Vkx6UEOhNbawmPrDtzF4vXViCdHbfUTcpd2qvmXgVlTg7cULSw4MzGdN-Uqbp6-MnpvGIFrRVOVooRE5u8zhrbRcZL4RApjr9SrIEPm1WSp7Qlj8wjktBL4K1bNKn4NE9-AFtOu_0X-lL0Afav41RxxhqQyL_Ox3o3YI8Y.hz022ycDLUciahf-YOeEDw/inceptionresnetv2-520b38e4.pth"
PY3=python
SRC=$(wildcard */**.py)

all: $(SRC)
	-git push
	[ -f ./cc-test-reporter ] || curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > ./cc-test-reporter
	chmod +x ./cc-test-reporter
	./cc-test-reporter before-build
	-coverage run -m pytest .
	coverage xml
	./cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT
push: $(SRC)
	git push # push first as kernel will download the codes, so put new code to github first
	eval 'echo $$(which $(PY3)) is our python executable'
	$(PY3) -m pytest -s -k "TestCo" tests/test_coord.py # && cd .runners/intercept-resnet-384/ && $(PY3) main.py
lint: $(SRC)
	echo $(SRC)
	pylint -E $(SRC)
lstm:
	-git stash; git pull
	-python lstm.py 2>&1
	bash -c 'while true; do test x$$(git pull | grep -c Already) = x1 || python lstm.py 2>&1; sleep 10; echo -n .; done'
test: $(SRC)
	eval 'echo $$(which $(PY3)) is our python executable'
	$(PY3) -m pytest -k "test_generate_runner" tests/test_coord.py; cd .runners/intercept-resnet-384/ && $(PY3) main.py
clean:
	-rm -rf __pycache__ mylogs dist/* build/*
submit:
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation
run_submit:
	python DAF3D/Train.py
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation
twine:
	python3 -m twine -h >/dev/null || ( echo "twine not found, will install it." ; python3 -m pip install --user --upgrade twine )
publish: twine
	if [ x$(TAG) = x ]; then echo "Please pass TAG flag when you call make"; false; else git tag -s $(TAG); fi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
install_dep:
	#mkdir -p /root/.cache/torch/checkpoints; wget  $(URL) && cp inceptionresnetv2-520b38e4.pth /root/.cache/torch/checkpoints/inceptionresnetv2-520b38e4.pth
	test -z "$(python3 -m albumentations 2>&1 | grep direct)" && pip install -U git+https://github.com/albu/albumentations
	test -z "$(python3 -m segmentation_models_pytorch 2>&1 | grep direct)" && pip install git+https://github.com/qubvel/segmentation_models.pytorch
connect:
	stty raw -echo && ( ps aux | sed -n 's/.*vvlp \([0-9]\{1,\}\)/\1/p' | xargs -I{} ncat 127.1 {} )


.PHONY: clean connect
