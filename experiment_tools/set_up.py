import datetime as dt
import logging
import os
import platform
import subprocess

from experiment_tools.set_random_seed import fix_seed
from experiment_tools.start_logging import get_logger


def get_directory(cfg: dict, date_time: dt.datetime) -> dict:
    """
    ディレクトリを取得する関数

    Parameters
    ----------
    cfg: dict
        config データ
    date_time: dt.datetime
        日付と時間のデータ

    Returns
    ----------
    cfg: dict
        更新された config データ
    """
    # 出力ディレクトリのパスを作成
    output_dir = "outputs"
    phase_type = cfg.get("phase_type", "experiment")
    phase_output_dir = os.path.join(output_dir, phase_type)

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(phase_output_dir):
        os.makedirs(phase_output_dir, exist_ok=True)

    # 日付と時間のディレクトリを作成
    date = str(date_time.date())
    date_dir = os.path.join(phase_output_dir, date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir, exist_ok=True)

    # 時間のディレクトリを作成
    time = str(date_time.time()).split(".")[0].replace(":", "-")
    time_dir = os.path.join(date_dir, time)
    os.makedirs(time_dir, exist_ok=True)

    # config データを更新
    cfg["log"]["log_file"] = os.path.join(time_dir, cfg["log"]["log_file"])

    return cfg


def get_git_info(logger: logging.Logger) -> logging.Logger:
    """
    Git リポジトリ情報をログに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        ロガーオブジェクト

    Returns
    ----------
    logger: logging.Logger
        更新されたロガーオブジェクト
    """
    # commit id を取得
    git_info = "commit id: "
    git_info += subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

    # username を取得
    git_name = "username: "
    git_name += subprocess.check_output(
        ["git", "config", "user.name"]
    ).decode().strip()

    # ログに記録
    logger.info(git_info)
    logger.info(git_name)

    return logger


def get_os_info(logger: logging.Logger) -> logging.Logger:
    """
    OS 情報をログに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        ロガーオブジェクト

    Returns
    ----------
    logger: logging.Logger
        更新されたロガーオブジェクト
    """
    # OS 情報を取得
    os_info = "\n\n"
    spec = [
        f"\tOS: {platform.system()} {platform.release()}",
        f"\tProcessor: {platform.processor()}",
        f"\tMachine: {platform.machine()}",
        f"\tNode: {platform.node()}",
        f"\tPython Version: {platform.python_version()}",
    ]
    for item in spec:
        os_info += item
        os_info += "\n"

    # ログに記録
    logger.info(f"OS information: {os_info}")

    return logger


def get_pip_list(logger: logging.Logger) -> logging.Logger:
    """
    インストール済み Python パッケージをログに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        ロガーオブジェクト

    Returns
    ----------
    logger: logging.Logger
        更新されたロガーオブジェクト
    """
    # pip list を取得
    pip_list = "pip3 list:\n\n"
    pip_list += subprocess.check_output(
        ["pip3", "list"]
    ).decode().strip()
    pip_list += "\n"

    # ログに記録
    logger.info(pip_list)

    return logger


def start_experiment(cfg: dict) -> logging.Logger:
    """
    実験を開始する関数

    Parameters
    ----------
    cfg: dict
        config データ

    Returns
    ----------
    logger: logging.Logger
        設定済みロガーオブジェクト
    """
    # 日付と時間を取得し, ディレクトリを設定
    date_time = dt.datetime.now()
    cfg = get_directory(cfg, date_time)

    # ロガーを設定
    logger = get_logger(cfg)

    # 乱数シードを固定
    fix_seed(cfg["seed"])

    # Git 情報, OS 情報, pip list をログに記録
    logger = get_git_info(logger)
    logger = get_os_info(logger)
    logger = get_pip_list(logger)

    return logger
