import difflib
import json
import logging


def get_config(default: str, experiment: str) -> tuple[dict, dict, str, str]:
    """
    デフォルトと実験用の設定ファイルを読み込み, 辞書型および文字列型で返す関数

    Parameters
    ----------
    default: str
        デフォルト設定ファイルのパス
    experiment: str
        実験用設定ファイルのパス

    Returns
    ----------
    default_cfg: dict
        辞書型のデフォルト config
    exp_cfg: dict
        辞書型の実験用 config
    default_cfg_str: str
        文字列型のデフォルト config
    exp_cfg_str: str
        文字列型の実験用 config
    """
    # デフォルトのデータを取得し文字列型に変換
    with open(default) as f:
        default_cfg = json.load(f)
    default_cfg_str = json.dumps(default_cfg, indent=4)

    # 実験用のデータを取得し文字列型に変換
    with open(experiment) as f:
        exp_cfg = json.load(f)
    exp_cfg_str = json.dumps(exp_cfg, indent=4)

    return default_cfg, exp_cfg, default_cfg_str, exp_cfg_str


def get_diff(
    default: str, exp: str, logger: logging.Logger
) -> logging.Logger:
    """
    デフォルトと実験用の config の差分を取得し, ロガーに出力する関数

    Parameters
    ----------
    default: str
        文字列型のデフォルト config
    exp: str
        文字列型の実験用 config
    logger: logging.Logger
        ロガーオブジェクト

    Returns
    ----------
    logger: logging.Logger
        差分を出力したロガーオブジェクト
    """
    # 差分を取得
    differ = difflib.ndiff(
        default.splitlines(keepends=True), exp.splitlines(keepends=True)
    )

    # 差分のうち, 追加された行(+)と削除された行(-)を抽出
    diff_parts = "\n"
    for line in differ:
        if line.startswith("+") or line.startswith("-"):
            diff_parts += "\n" + line.strip()
    diff_parts += "\n"

    # ロガーに差分を出力
    logger.info(f"diff: {diff_parts}")

    return logger
