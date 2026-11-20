# PricePilot Makefile

.PHONY: all pipeline test dashboard clean help

help:
	@echo "Available commands:"
	@echo "  make pipeline   - Run full pipeline end-to-end (stages 1 to 6)"
	@echo "  make test       - Execute full pytest suite"
	@echo "  make dashboard  - Launch interactive Streamlit dashboard"
	@echo "  make clean      - Remove cache and compiled files"

pipeline:
	python3 scripts/run_pipeline.py

test:
	pytest tests/

dashboard:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
