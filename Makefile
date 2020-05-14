export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)
export CC_TEST_REPORTER_ID := 501f2d3f82d0d671d4e2dab422e60140a9461aa51013ecca0e9b2285c1b4aa43

URL="https://www.kaggleusercontent.com/kf/33958071/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..dOPYcZgFMrXYOqS_uN2DGg.mFYvzee-tM5ti6imydrGvHkGP40aHRAMb4gU_SsIIJ7gsFab4J-Aa059GhM2cOsNGYelaKb1JZrbnFc9dxf3sifSTJe-KZkZSLBQgd9uqQsaKN6lvt1jrrNjqO3dFlUl_dJPpdJ5fkpVMyTbN1XZCaFo7LLN1JqjUch3Tamox79hLahZeeGtxLJieB8sjhFy4iGKiXQGoQFpy641Wtse3QjVrt5V1G1q8xBDq9-x2aW_3Gv6syem666TnPF8gTwjNo2CM0AaN2AO4MlPImK5XUKbDQx8rTEGprIl-J4dcMeosD1Iqt3ZT6wvFo4uNn6Ob9sQJKScXxNs5BojJHlCC7WOYJU_3Wm1wEnSnbi95e1nTrUbyTjptBCd1ksRIM3WXqNSvN-NUeykq40u6s3uz-OTSwro5-Wow0vuSQp9xiNXAGs5Qafb1PGloe9kujSZOZRzwpL3FUoezM7qns3YqAjyjryl7-T2W_dNlL4hA_XhvDCl513TNQX4QYD8wZk6PU3QXUCK3JGqr4-NJgquWnf3SddEtt1F1-8N6Zkh95VS2OR3VgsSEA7HwFg20675JvWQ37EFgk7BAAu92lgh9LhvxaG0rbvAWfK4ICsLQzVbqdAOgxj7VEpSqT78Ge1x3mGlqhdo2RzbwQG4Itj49q0p7EQ6li43rqf_-hRVamU7EVxENu06DW_T65QTgCt0.Afj-QzF4tNc5Jtq1cKTAFQ/inceptionresnetv2-520b38e4.pth"
PY3=python
SRC=$(wildcard *.py)

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
test: $(SRC)
	eval 'echo $$(which $(PY3)) is our python executable'
	$(PY3) -m pytest -k "TestCo" tests/test_coord.py && cd .runners/intercept-resnet-384/ && $(PY3) main.py
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
	mkdir -p /root/.cache/torch/checkpoints; wget  $(URL) && cp inceptionresnetv2-520b38e4.pth /root/.cache/torch/checkpoints/inceptionresnetv2-520b38e4.pth
	bash -c "[ -z $(python3 -m albumentations | grep direct) ] && pip install -U git+https://github.com/albu/albumentations"
	bash -c "[ -z $(python3 -m segmentation_models_pytorch | grep direct) ] && pip install git+https://github.com/qubvel/segmentation_models.pytorch"


.PHONY: clean
