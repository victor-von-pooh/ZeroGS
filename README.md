# ZeroGS

3D Gaussian Splatting を PyTorch でフルスクラッチ実装するプロジェクト。

## 背景と目的

[GNN_Colmap](https://github.com/victor-von-pooh/GNN_Colmap) は、COLMAP の希薄復元（SfM）を GNN で置き換えることで、カメラ姿勢と 3D 点群を推定する。
ZeroGS は、その **GNN_Colmap の出力がどれくらい良いのかを定量的に評価する** ために、3DGS による Novel View Synthesis パイプラインを提供する。

具体的には、GNN_Colmap が出力する COLMAP 互換フォーマット（cameras.txt / images.txt / points3D.txt）を入力として受け取り、3DGS で学習・レンダリングし、PSNR / SSIM / LPIPS で評価する。

## 設計方針

- 依存ライブラリは **PyTorch + 標準的な科学計算ライブラリ**（numpy, scipy, PIL 等）のみ
- gsplat, nerfstudio 等の 3DGS 専用ライブラリは使用しない
- ラスタライザも PyTorch で自前実装し、アーキテクチャレベルでカスタマイズ可能にする

## 入力仕様（GNN_Colmap の出力フォーマット）

| ファイル | 内容 |
|---|---|
| `sparse/0/cameras.txt` | カメラ内部パラメータ（SIMPLE_RADIAL: f, cx, cy, k） |
| `sparse/0/images.txt` | カメラ外部パラメータ（QW, QX, QY, QZ, TX, TY, TZ） |
| `sparse/0/points3D.txt` | 3D 点群（X, Y, Z, R, G, B=128, ERROR=0, TRACK=空） |
| `images/` | 入力画像群 |

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