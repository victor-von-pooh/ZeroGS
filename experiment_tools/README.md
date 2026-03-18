# experiment_tools

3D Gaussian Splatting の学習における実験管理を行うサブモジュール。
実験の再現性確保・ログ記録・環境情報の保存を一括で提供する。

## モジュール構成

| ファイル | 概要 |
|---|---|
| `set_up.py` | 実験セットアップの統合エントリポイント |
| `set_random_seed.py` | 乱数シードの固定 |
| `start_logging.py` | ロガーの初期化 |

## 使い方

`start_experiment(cfg)` を呼び出すことで、以下の処理が自動的に実行される。

```python
from experiment_tools.set_up import start_experiment

logger = start_experiment(cfg)
```

### `start_experiment` が行う処理

1. **出力ディレクトリの作成** — `outputs/<phase_type>/<date>/<time>/` の階層構造で実験ごとのディレクトリを生成
2. **ロガーの初期化** — 生成したディレクトリ内にログファイルを作成し、`logging.Logger` を返却
3. **乱数シードの固定** — `random`, `numpy`, `torch`（CPU/CUDA）のシードを統一的に設定
4. **環境情報のログ記録** — Git commit ID・ユーザ名、OS・Python バージョン、`pip3 list` の結果を記録

## config (`cfg`) の構造

```python
cfg = {
    "phase_type": "experiment",  # 出力ディレクトリのサブフォルダ名
    "seed": 42,                  # 乱数シード
    "log": {
        "log_file": "experiment.log",      # ログファイル名
        "log_formatter": "%(asctime)s ...", # ログフォーマット文字列
    },
}
```

## 各モジュールの詳細

### `set_random_seed.py`

- `fix_seed(seed: int)` — `random`, `numpy`, `torch`（`manual_seed`, `cuda.manual_seed_all`）のシードを固定し、`cudnn.deterministic = True` を設定

### `start_logging.py`

- `get_logger(cfg: dict)` — `cfg["log"]` の設定に基づき、`FileHandler` 付きの `logging.Logger`（INFO レベル）を返却

### `set_up.py`

- `get_directory(cfg, date_time)` — 日時ベースの出力ディレクトリを作成し、`cfg["log"]["log_file"]` のパスを更新
- `get_git_info(logger)` — Git commit ID とユーザ名をログに記録
- `get_os_info(logger)` — OS、プロセッサ、マシン、ノード名、Python バージョンをログに記録
- `get_pip_list(logger)` — `pip3 list` の出力をログに記録
- `start_experiment(cfg)` — 上記すべてを統合して実行するエントリポイント
