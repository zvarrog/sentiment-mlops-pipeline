-- Инициализация базы данных metrics для мониторинга производительности

-- Таблица для хранения метрик длительности задач
CREATE TABLE IF NOT EXISTS task_metrics (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    duration_sec NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'success',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Индексы для быстрого поиска
    CONSTRAINT unique_task_execution UNIQUE (dag_id, task_id, execution_date)
);

-- Индексы для эффективных запросов
CREATE INDEX IF NOT EXISTS idx_task_metrics_dag_id ON task_metrics(dag_id);
CREATE INDEX IF NOT EXISTS idx_task_metrics_task_id ON task_metrics(task_id);
CREATE INDEX IF NOT EXISTS idx_task_metrics_execution_date ON task_metrics(execution_date);
CREATE INDEX IF NOT EXISTS idx_task_metrics_created_at ON task_metrics(created_at);

-- Таблица для метрик моделей
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC(10, 6) NOT NULL,
    split VARCHAR(50) NOT NULL, -- train/val/test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для метрик моделей
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_execution_date ON model_metrics(execution_date);

-- Предоставляем права на таблицы
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
