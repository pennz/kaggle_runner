export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43
PY3=python3
SRC=$(wildcard *.py)

all: push $(SRC)
	cc-test-reporter before-build
	-coverage run -m pytest .
	coverage xml
	cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT
push: $(SRC)
	git push # push first as kernel will download the codes, so put new code to github first
	$(PY3) -m pytest -k "TestCo" # && cd .runners/intercept-resnet-384/ && $(PY3) main.py
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

.PHONY: clean
