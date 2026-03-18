import matplotlib.pyplot as plt


def plot_training_curve(train_loss_list: list, output_path: str) -> None:
    """
    学習曲線をプロットして保存する関数

    Parameters
    ----------
    train_loss_list: list
        学習過程の損失を格納したリスト
    output_path: str
        プロット画像の保存先パス

    Returns
    -------
    None
    """
    # エポック数のリストを作成
    epochs = list(range(1, len(train_loss_list) + 1))

    # 出力画像の設定
    plt.figure(figsize=(18, 12), tight_layout=True)
    plt.title("Training Loss over Epochs", size=15, color="red")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 学習曲線のプロットと凡例の表示
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    # プロットの保存
    plt.savefig(output_path)
    plt.close()
