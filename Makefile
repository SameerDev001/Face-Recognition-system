# ---------------------------------------------------------------------------
# Face Recognition System — Pipeline Makefile
# ---------------------------------------------------------------------------

PYTHON = py
PIP = $(PYTHON) -m pip

.PHONY: help install collect train run clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install project dependencies"
	@echo "  make collect  - Start face data collection"
	@echo "  make train    - Train the LBPH recognition model"
	@echo "  make run      - Run real-time face recognition"
	@echo "  make clean    - Remove datasets and trained models"

install:
	$(PIP) install -r requirements.txt

collect:
	$(PYTHON) src/data_collection.py

train:
	$(PYTHON) src/train_model.py

run:
	$(PYTHON) src/recognize.py

clean:
	@echo "Cleaning dataset and models..."
	if exist dataset (rmdir /s /q dataset && mkdir dataset)
	if exist models (rmdir /s /q models && mkdir models)
	@echo "Cleanup complete."
