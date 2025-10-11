-- ВАЖНО: CREATE DATABASE нельзя выполнять внутри транзакции.
-- Используем psql-команду \gexec, чтобы выполнить CREATE DATABASE как отдельный оператор.
SELECT 'CREATE DATABASE airflow_meta'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'airflow_meta'
)\gexec
