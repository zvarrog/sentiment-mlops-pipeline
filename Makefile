.PHONY: help install test lint format clean docker-up docker-down

help:
	@echo "Доступные команды:"
	@echo "  make install       - установка зависимостей"
	@echo "  make test          - запуск unit тестов"
	@echo "  make test-all      - запуск всех тестов (включая интеграционные)"
	@echo "  make lint          - проверка кода (ruff)"
	@echo "  make format        - форматирование кода (black + isort)"
	@echo "  make clean         - очистка временных файлов"
	@echo "  make docker-up     - запуск Docker сервисов"
	@echo "  make docker-down   - остановка Docker сервисов"
	@echo "  make streamlit     - запуск Streamlit demo"
	@echo "  make validate-dag  - проверка синтаксиса DAG"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements.api.txt
	pip install -r tests/requirements.txt
	pip install ruff black isort pre-commit

test:
	pytest tests/test_core_modules.py -v -m "not integration"

test-all:
	pytest tests/ -v

test-integration:
	pytest tests/test_integration.py -v -m integration

test-coverage:
	pytest tests/test_core_modules.py --cov=scripts --cov-report=html --cov-report=term

lint:
	ruff check scripts/ tests/ --exit-zero
	black --check scripts/ tests/ --line-length 88

format:
	isort scripts/ tests/
	black scripts/ tests/ --line-length 88
	ruff check scripts/ tests/ --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

docker-up:
	docker-compose up -d
	@echo "Ожидание готовности сервисов..."
	sleep 10
	@echo "Сервисы запущены:"
	@echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
	@echo "  - MLflow UI: http://localhost:5000"
	@echo "  - FastAPI: http://localhost:8000"
	@echo "  - Prometheus: http://localhost:9090"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f --tail=100

streamlit:
	streamlit run streamlit_app.py

validate-dag:
	python -m py_compile airflow/dags/kindle_unified_pipeline.py
	@echo "✓ DAG синтаксис корректен"

pre-commit-install:
	pre-commit install
	@echo "✓ Pre-commit hooks установлены"

pre-commit-run:
	pre-commit run --all-files
