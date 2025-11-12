#!/bin/bash
# Скрипт для создания нескольких БД в PostgreSQL при инициализации контейнера
# Использует переменную POSTGRES_MULTIPLE_DATABASES (разделенные запятыми названия БД)

set -e
set -u

function create_database() {
    local database=$1
    
    # Whitelist: только буквы, цифры и подчёркивания
    if [[ ! "$database" =~ ^[a-zA-Z0-9_]+$ ]]; then
        echo "ОШИБКА: Недопустимое имя БД '$database' (только a-zA-Z0-9_)" >&2
        return 1
    fi
    
    echo "Создание базы данных '$database'"
    
    # Идентификатор через psql -c для безопасного экранирования
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE "$database"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
        GRANT ALL PRIVILEGES ON DATABASE "$database" TO "$POSTGRES_USER";
EOSQL
}

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Инициализация нескольких баз данных: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        create_database "$db"
    done
    echo "Создание нескольких БД завершено"
else
    echo "POSTGRES_MULTIPLE_DATABASES не задана — пропуск создания дополнительных БД"
fi
