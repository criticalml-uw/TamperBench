VERSION ?= 0.0.1
FULL_IMAGE_NAME = ghcr.io/tomtseng/tamperbench:$(VERSION)

.PHONY: docker-build devbox cpu large

docker-build:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Warning: Uncommitted changes or untracked files found:"; \
		git status --short; \
		read -p "Do you want to continue? (y/n) " -n 1 -r; \
		echo; \
		if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
			echo "Build cancelled."; \
			exit 1; \
		fi; \
	fi
	docker build . --platform=linux/amd64 --tag=$(FULL_IMAGE_NAME)
	docker push $(FULL_IMAGE_NAME)

# Devbox env variables. Can be overridden, e.g., `GPU=4 make devbox` makes a
# 4-GPU devbox rather than a 1-GPU devbox.
COMMIT_HASH ?= $(shell git rev-parse HEAD)
CPU ?= 1
MEMORY ?= 30G
GPU ?= 1
# By default, kill the devbox after some time so that it doesn't
# unintentionally use up cluster GPUs.
SLEEP_TIME ?= 2d
USER ?= $(shell whoami)
DEVBOX_NAME ?= devbox-$(USER)
YAML_FILE ?= k8s/devbox.yaml

# Make and launch a devbox
devbox:
	git push
	python -c "print(open('${YAML_FILE}').read().format( \
                NAME='${DEVBOX_NAME}', \
                USER='${USER}', \
                IMAGE='${FULL_IMAGE_NAME}', \
                COMMIT_HASH='${COMMIT_HASH}', \
                CPU='${CPU}', \
                MEMORY='${MEMORY}', \
                GPU='${GPU}', \
                SLEEP_TIME='${SLEEP_TIME}', \
        ))" | kubectl create --namespace u-tamperbench --filename -

# Make a CPU-only devbox with `make cpu devbox`
cpu:
	$(eval SLEEP_TIME := 30d)
	$(eval GPU := 0)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-cpu)

# Make a 2-GPU devbox with `make large devbox`
large:
	$(eval MEMORY := 80G)
	$(eval GPU := 2)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-large)

# Tom's `whoami` is vague, so override it with something more legible.
tom:
	$(eval USER := tomtseng)
