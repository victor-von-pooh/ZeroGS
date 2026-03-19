# src/gs

3D Gaussian Splatting のモデル定義と学習スクリプト。

参考論文: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.14737) (Kerbl et al., 2023)

## ファイル一覧

| ファイル | 概要 |
|---|---|
| `model.py` | GaussianModel クラスの定義。Gaussian パラメータの保持、レンダリング、ADC を実装 |
| `train.py` | 学習のエントリポイント。データ読み込みから学習・評価・エクスポートまでを実行する |
