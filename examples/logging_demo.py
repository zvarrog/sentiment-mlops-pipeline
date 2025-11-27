import logging
import logging.config


# Эмуляция конфигурации из logging_config.py
def setup_demo_logging():
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "[%(levelname)s] %(name)s: %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            # Логгер для всего пакета scripts
            "scripts": {"level": "INFO", "handlers": ["console"], "propagate": False},
            # Специально настроим scripts.api_service, чтобы показать иерархию
            "scripts.api_service": {
                "level": "WARNING",  # Глушим INFO для этого модуля
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
    logging.config.dictConfig(config)


def run_demo():
    setup_demo_logging()
    print("--- DEMO START ---\n")

    # 1. Использование __name__ (правильный путь)
    # Представим, что мы внутри scripts/api_service.py
    logger_name_module = "scripts.api_service"
    log_module = logging.getLogger(logger_name_module)

    print(f"1. Логгер через __name__ ('{logger_name_module}')")
    print(
        f"   Настройки наследуются от 'scripts' или переопределены для '{logger_name_module}'"
    )
    log_module.info("Это INFO сообщение (не должно показаться, т.к. level=WARNING)")
    log_module.warning("Это WARNING сообщение (должно показаться)")

    print("\n" + "-" * 20 + "\n")

    # 2. Использование кастомного имени (твой вариант)
    logger_name_custom = "api_service"
    log_custom = logging.getLogger(logger_name_custom)

    print(f"2. Логгер через строку ('{logger_name_custom}')")
    print("   Это отдельный логгер, он не знает про настройки 'scripts'")
    print("   Он падает в 'root' логгер, если не настроен явно.")
    log_custom.info("Это INFO сообщение (покажется, т.к. root level=INFO)")
    log_custom.warning("Это WARNING сообщение (покажется)")

    print("\n--- DEMO END ---")


if __name__ == "__main__":
    run_demo()
