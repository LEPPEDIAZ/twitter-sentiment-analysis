PORT ?= 8000
CONTAINER := ana_q_aylien
VERSION ?= `cat VERSION`
NUM_WORKERS ?= 1
DEMO_NAME ?= new-demo
LOG_LEVEL ?= DEBUG


.PHONY: dev
dev:
	pip install -e .
	pip install -r requirements.txt


.PHONY: test
test:
	pytest ana_q_aylien


.PHONY: demo
demo:
	@python -m quickstarter.create_demo --name $(DEMO_NAME)
	make run --directory demos/$(DEMO_NAME)


# run service locally
.PHONY: run
run:
	gunicorn -b:$(PORT) 'ana_q_aylien.serving:run_app()' --log-level=$(LOG_LEVEL)


# service must be running, i.e. start "make run" first
.PHONY: example-request-predict
example-request-predict:
	curl -i -H "Content-Type: application/json" -X POST --data "@./examples/example_request.json" "http://localhost:8000/predict"


# build docker container
.PHONY: build
build:
	docker build --no-cache -t $(CONTAINER):$(VERSION) -f Dockerfile .
	@echo "To run the container, use:"
	@echo "docker run -p 8000:8000 -e --rm -it $(CONTAINER):$(VERSION)"
