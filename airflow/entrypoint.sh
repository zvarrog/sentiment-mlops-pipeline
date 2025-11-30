#!/bin/bash
# Entrypoint для Airflow контейнера
# Выполняет инициализацию БД, создание пользователя и запуск сервисов.

set -e

echo "=== Инициализация Airflow ==="

# Создаём директории для данных
mkdir -p /opt/airflow/data/processed /opt/airflow/data/raw

# Устанавливаем права (fallback на chmod если chown недоступен)
chown -R 50000:0 /opt/airflow/data 2>/dev/null || chmod -R 777 /opt/airflow/data

# Удаляем stale PID файлы
rm -f /opt/airflow/airflow-webserver*.pid

# Миграция БД
echo "Миграция базы данных..."
airflow db migrate

# Создаём admin пользователя если не существует
if ! airflow users list | grep -q "admin@example.com"; then
    echo "Создание admin пользователя..."
    airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
fi

# Запускаем webserver в фоне
echo "Запуск Airflow webserver..."
airflow webserver -p 8080 &
WEBSERVER_PID=$!

# Ждём готовности webserver
echo "Ожидание запуска Airflow UI..."
for i in {1..30}; do
    if curl -sSf http://localhost:8080/health >/dev/null 2>&1; then
        echo "Airflow UI доступен: http://localhost:8080 (admin/admin)"
        break
    fi
    sleep 2
done

# Запускаем scheduler в foreground
echo "Запуск Airflow scheduler..."
exec airflow scheduler
