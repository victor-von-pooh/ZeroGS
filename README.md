# ZeroGS

3D Gaussian Splatting を PyTorch でフルスクラッチ実装するプロジェクト。

参考論文: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.14737) (Kerbl et al., 2023)

## 背景と目的

[GNN_Colmap](https://github.com/victor-von-pooh/GNN_Colmap) は、COLMAP の希薄復元（SfM）を GNN で置き換えることで、カメラ姿勢と 3D 点群を推定する。
ZeroGS は、その **GNN_Colmap の出力がどれくらい良いのかを定量的に評価する** ために、3DGS による Novel View Synthesis パイプラインを提供する。

具体的には、GNN_Colmap が出力する COLMAP 互換フォーマット（cameras.txt / images.txt / points3D.txt）を入力として受け取り、3DGS で最適化・レンダリングし、PSNR / SSIM / LPIPS で評価する。

## 設計方針

- 依存ライブラリは **PyTorch + 標準的な科学計算ライブラリ**（numpy, scipy, PIL 等）のみ
- gsplat, nerfstudio 等の 3DGS 専用ライブラリは使用しない
- ラスタライザも PyTorch で自前実装し、アーキテクチャレベルでカスタマイズ可能にする
- ラスタライザは `torch.autograd.Function` によるカスタム backward で実装し、Gaussian ごとのバウンディングボックス内だけを再計算することでメモリ使用量を O(N × パッチサイズ²) に抑える
- ラスタライザは Python for-loop ベースのため、MPS / CUDA 環境でもカーネル起動オーバーヘッドを避けるために CPU で実行する。デバイス `"auto"` では CUDA → CPU の順で選択する

## ディレクトリ構成

```
ZeroGS/
├── configs/                # ハイパーパラメータ管理
│   ├── default/            #   デフォルト値（基準値、Git 追跡対象）
│   │   └── gs.json
│   └── experiment/         #   実験用（default/ からコピーして使用、Git 追跡対象外）
├── experiment_tools/       # 実験管理サブモジュール
│   ├── set_up.py           #   実験セットアップの統合エントリポイント
│   ├── set_random_seed.py  #   乱数シード固定
│   └── start_logging.py    #   ロガー初期化
├── data/                   # 学習データ（GNN_Colmap の出力、Git 追跡対象外）
│   ├── images/             #   入力画像群
│   └── sparse/0/           #   COLMAP 互換フォーマット（txt / bin 両対応）
│       ├── cameras.txt (or cameras.bin)
│       ├── images.txt  (or images.bin)
│       └── points3D.txt (or points3D.bin)
├── outputs/                # 学習結果の出力先（Git 追跡対象外）
│   └── gs/<date>/<time>/   #   モデル・PLY・レンダリング画像など
├── src/                    # ソースコード
│   ├── gs/                 #   3DGS 固有のモデル定義・学習スクリプト
│   │   ├── model.py        #     GaussianModel（レンダリング・ADC・SH）
│   │   └── train.py        #     学習実行のメインスクリプト
│   └── utils/              #   ドメインモデルに依存しない汎用ユーティリティ
│       ├── cfg_diff.py     #     config の読み込みと差分ログ出力
│       ├── loss.py         #     損失関数（L1・SSIM・Combined Loss）
│       ├── preprocessing.py    # COLMAP パーサー、画像読み込み、初期色推定
│       ├── result.py       #     評価メトリクス、PLY エクスポート、レンダリング画像保存
│       └── trainer.py      #     Optimizer 選択、学習ループ
└── venv/                   # Python 仮想環境
```

## src — ソースコード

GNN_Colmap の出力を入力として、3DGS で最適化・レンダリング・評価を行う。

1. **前処理** — COLMAP 形式のファイルをパースし、画像を読み込み、3D 点群に対して初期色を推定する。画像は最適化用と評価用に分離する
2. **最適化** — L1 + SSIM の損失関数を最小化しながら Gaussian パラメータ（位置・SH 係数・不透明度・スケール・回転）を最適化する。Adaptive Density Control で Gaussian の追加・分割・削除を行う
3. **評価・エクスポート** — 評価用画像に対して PSNR / SSIM / LPIPS を計算し、標準 3DGS PLY 形式で Gaussian をエクスポートする

詳細は [src/README.md](src/README.md) を参照。

## configs — ハイパーパラメータ管理

- `configs/default/` にデフォルトのハイパーパラメータを JSON で保持する（7000 イテレーションの本番設定）
- 実験時は `default/` から `experiment/` へ config ファイルをコピーし、パラメータを変更して使用する
- `default/` は Git 追跡対象、`experiment/` は追跡対象外
- 動作確認・速度確認目的の短期実験では、300 イテレーション用の ADC パラメータ調整を行う（詳細は [configs/experiment/README.md](configs/experiment/README.md) を参照）

詳細は [configs/README.md](configs/README.md) を参照。

## experiment_tools — 実験管理サブモジュール

`start_experiment(cfg)` を呼び出すことで、以下を一括実行する。

1. 日時ベースの出力ディレクトリ作成（`outputs/<phase_type>/<date>/<time>/`）
2. ロガーの初期化
3. 乱数シードの固定（`random`, `numpy`, `torch`）
4. 環境情報のログ記録（Git 情報、OS 情報、pip list）

詳細は [experiment_tools/README.md](experiment_tools/README.md) を参照。

## data — 学習データ

GNN_Colmap の出力（COLMAP 互換フォーマット）と入力画像群を `data/` に配置する。txt 形式・bin 形式のどちらでも動作し、bin 形式の場合は学習開始時に自動的に txt へ変換される。

詳細は [data/README.md](data/README.md) を参照。

## outputs — 学習結果

学習実行時に `outputs/gs/<日付>/<時刻>/` へ以下が自動保存される。

- `gs.log` — 実験ログ
- `training_curve.png` — 学習曲線のプロット
- `final_model.pth` — 学習済みモデルの重み
- `gaussians.ply` — 標準 3DGS PLY 形式の Gaussian パラメータ
- `renders/` — 評価用画像のレンダリング結果（正解画像との比較）

詳細は [outputs/README.md](outputs/README.md) を参照。

## セットアップから実行まで

### 1. 仮想環境の構築

仮想環境を立ち上げる。

```bash
python3 -m venv venv
source venv/bin/activate
```

必要なライブラリをインストールする。

```bash
pip3 install -U pip
pip3 install -r requirements.txt
```

### 2. 学習データの準備

GNN_Colmap の学習・推論を実行し、出力された `sparse/0/` と入力画像 `images/` を `data/` に配置する。txt 形式・bin 形式のどちらでも動作する。

```
data/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── sparse/0/
    ├── cameras.txt  # または cameras.bin（起動時に自動変換）
    ├── images.txt   # または images.bin
    └── points3D.txt # または points3D.bin
```

### 3. 実験用 config の作成

`configs/default/gs.json` を `configs/experiment/gs.json` にコピーし、必要に応じてパラメータを変更する。

```bash
cp configs/default/gs.json configs/experiment/gs.json
```

### 4. 学習の実行

プロジェクトルートから以下を実行する。

```bash
python3 -m src.gs.train
```

### 5. 結果の確認

学習結果は `outputs/gs/<日付>/<時刻>/` に出力される。

- `gs.log` — 実験ログ（データ統計、損失統計、PSNR / SSIM / LPIPS）
- `training_curve.png` — 学習曲線のプロット
- `final_model.pth` — 学習済みモデルの重み
- `gaussians.ply` — 標準 3DGS PLY 形式の Gaussian パラメータ
- `renders/` — 評価用画像ごとに正解（左）とレンダリング（右）を並べた画像

## Commit のルール

Commit の際は以下のルールに合わせて種類ごとにする。

🎉 初めてのコミット (Initial Commit)  
🔖 バージョンタグ (Version Tag)  
✨ 新機能 (New Feature)  
🐛 バグ修正 (Bugfix)  
♻️ リファクタリング (Refactoring)  
📚 ドキュメント (Documentation)  
🎨 デザインUI/UX (Accessibility)  
🐎 パフォーマンス (Performance)  
🔧 ツール (Tooling)  
🚨 テスト (Tests)  
💩 非推奨追加 (Deprecation)  
🗑️ 削除 (Removal)  
🚧 WIP (Work In Progress)