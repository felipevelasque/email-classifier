import logging, os, sys

def setup_logger():
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    handler = logging.StreamHandler(sys.stdout)
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    if fmt == "json":
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        except Exception:
            # fallback para texto se lib não estiver instalada
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
