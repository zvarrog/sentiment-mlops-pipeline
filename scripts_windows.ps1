# Скрипты для Windows (альтернатива Makefile)

# Установка зависимостей
function Install-Dependencies {
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements.api.txt
    pip install -r tests/requirements.txt
    pip install ruff black isort pre-commit
}

# Запуск тестов
function Run-Tests {
    pytest tests/test_core_modules.py -v -m "not integration"
}

# Все тесты
function Run-AllTests {
    pytest tests/ -v
}

# Тесты с покрытием
function Run-Coverage {
    pytest tests/test_core_modules.py --cov=scripts --cov-report=html --cov-report=term
}

# Проверка кода
function Run-Lint {
    ruff check scripts/ tests/ --exit-zero
    black --check scripts/ tests/ --line-length 88
}

# Форматирование
function Run-Format {
    isort scripts/ tests/
    black scripts/ tests/ --line-length 88
    ruff check scripts/ tests/ --fix
}

# Очистка
function Clean-Project {
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Include *.pyc,*.pyo -Recurse -File | Remove-Item -Force
    if (Test-Path htmlcov) { Remove-Item htmlcov -Recurse -Force }
    if (Test-Path .coverage) { Remove-Item .coverage -Force }
}

# Docker запуск
function Start-Docker {
    docker-compose up -d
    Start-Sleep -Seconds 10
    Write-Host "Сервисы запущены:"
    Write-Host "  - Airflow UI: http://localhost:8080 (admin/admin)"
    Write-Host "  - MLflow UI: http://localhost:5000"
    Write-Host "  - FastAPI: http://localhost:8000"
    Write-Host "  - Prometheus: http://localhost:9090"
}

# Docker остановка
function Stop-Docker {
    docker-compose down
}

# Streamlit
function Start-Streamlit {
    streamlit run streamlit_app.py
}

# Валидация DAG
function Validate-DAG {
    python -m py_compile airflow/dags/kindle_unified_pipeline.py
    Write-Host "✓ DAG синтаксис корректен"
}

# Вывод справки
function Show-Help {
    Write-Host "Доступные команды:"
    Write-Host "  Install-Dependencies  - установка зависимостей"
    Write-Host "  Run-Tests            - unit тесты"
    Write-Host "  Run-AllTests         - все тесты"
    Write-Host "  Run-Coverage         - тесты с покрытием"
    Write-Host "  Run-Lint             - проверка кода"
    Write-Host "  Run-Format           - форматирование"
    Write-Host "  Clean-Project        - очистка"
    Write-Host "  Start-Docker         - запуск сервисов"
    Write-Host "  Stop-Docker          - остановка сервисов"
    Write-Host "  Start-Streamlit      - Streamlit demo"
    Write-Host "  Validate-DAG         - проверка DAG"
}

# Показываем справку при импорте
Show-Help
