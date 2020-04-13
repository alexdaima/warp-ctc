.PHONY: all
all: notice

notice:
	@echo 'You need to run `make <target>`'

login:
	docker login -u asciialex -p ${DOCKERHUB_ASCIIALEX_PASSWORD}
	
build:
	DOCKER_BUILDKIT=1 docker build -t asciialex/pytorch:1.4-cuda10.1-cudnn7-devel-warp-ctc . --no-cache

push: login
	docker push asciialex/pytorch:1.4-cuda10.1-cudnn7-devel-warp-ctc
