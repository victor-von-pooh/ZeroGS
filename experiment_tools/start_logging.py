import logging
from logging import FileHandler, Formatter, getLogger


def get_logger(cfg: dict) -> logging.Logger:
    """
    ロガーを取得する関数

    Parameters
    ----------
    cfg: dict
        config データ

    Returns
    ----------
    logger: logging.Logger
        設定されたロガーオブジェクト
    """
    # ロガーの設定
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = FileHandler(cfg["log"]["log_file"], mode="w")
    formatter = Formatter(cfg["log"]["log_formatter"])
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
