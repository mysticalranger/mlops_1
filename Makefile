.PHONY: setup train api dashboard test docker-build docker-run all monitor compare

setup:
    python -m pip install -r requirements.txt
    mkdir -p data models monitoring

preprocess:
    python scripts/preprocess.py

train:
    python scripts/train.py

monitor:
    python monitoring.py

api:
    python app.py

dashboard:
    python dashboard.py

test:
    python -m unittest discover tests

docker-build:
    docker build -t wellness-app .

docker-run:
    docker run -p 5000:5000 wellness-app

compare:
    python scripts/compare_models.py

all: preprocess train monitor api