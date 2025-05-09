# 画像インペイント処理アプリケーション

このアプリケーションは、指定されたオブジェクトを画像から自動的に検出し、その部分をインペイント（修復）するツールです。Grounding DINOとLaMaモデルを使用して、高品質な画像修復を実現します。

## 機能

- 画像内の特定のオブジェクト（車、人物など）の自動検出
- 検出されたオブジェクトのマスク生成
- LaMaモデルを使用した高品質なインペイント処理
- Gradioを使用した直感的なWebインターフェース

## 実装方法

<details>
<summary>Dockerを使用する場合</summary>

### 必要条件

- Docker
- Docker Compose
- NVIDIA GPU（推奨）とNVIDIA Container Toolkit

### インストール方法

1. リポジトリのクローン:
```bash
git clone https://github.com/aoyama-b/Make_Class_Inpaint_image.git
cd Make_Class_Inpaint_image
```

2. NVIDIA Container Toolkitのインストール（GPUを使用する場合）:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 使用方法

1. アプリケーションの起動:
```bash
docker-compose up --build
```

2. ブラウザで以下のURLにアクセス:
```
http://localhost:7860
```

</details>

<details>
<summary>ローカル環境で実装する場合</summary>

### 必要条件

- Python 3.9以上
- CUDA対応GPU（推奨）

### インストール方法

1. リポジトリのクローン:
```bash
git clone https://github.com/aoyama-b/Make_Class_Inpaint_image.git
cd Make_Class_Inpaint_image
```

2. 仮想環境の作成と有効化（推奨）:
```bash
# Anacondaを用いた例
conda create -n class_inpaint python=3.9
conda activate class_inpaint
```

3. 必要なパッケージのインストール:
```bash
pip install -r requirements.txt
```

### 使用方法

#### Gradioインターフェースを使用する場合

1. アプリケーションの起動:
```bash
python app.py
```

2. ブラウザで以下のURLにアクセス:
```
http://localhost:7860
```

#### コマンドラインから実行する場合

`make_inpaint_dataset.py`を使用して、フォルダ内の画像を一括処理できます：

```bash
python make_inpaint_dataset.py \
    -i {入力画像フォルダのパス} \
    -o {出力画像フォルダのパス} \
    -c {検出したいオブジェクトクラス} \
    -t {検出閾値（0.1-0.9）}
```

例：
```bash
python make_inpaint_dataset.py \
    -i ./input_images \
    -o ./output_images \
    -c car \
    -t 0.2
```

### 注意点

- GPUメモリを大量に使用するため、十分なGPUメモリを確保してください
- 初回実行時は、モデルのダウンロードに時間がかかる場合があります
- 処理時間は画像サイズとGPUの性能に依存します

</details>

<details>
<summary>Google Colabで実装する場合（実装が容易）</summary>

Colab用のノートブック（`inpaint_colab.ipynb`）を用意しています。以下の手順で実行できます：

1. リポジトリから`inpaint_colab.ipynb`をダウンロード
2. Google Colabでノートブックを開く
3. 各セルを順番に実行

詳細な手順はノートブック内に記載されています。

</details>

## 使用方法

- 入力画像をアップロード
- 対象クラス名を入力（例: "car", "person"）
- 検出閾値を調整（0.1-0.9）
- 「処理開始」ボタンをクリック

## 注意事項

- 初回起動時は、モデルのダウンロードに時間がかかる場合があります
- GPUメモリを大量に使用するため、十分なGPUメモリを確保してください
- 処理時間は画像サイズとGPUの性能に依存します

## 技術スタック

- Python 3.9
- PyTorch
- Transformers (Grounding DINO)
- Simple LaMa Inpainting
- Gradio
- Docker（オプション）

## 使用しているモデルとライブラリ

このプロジェクトでは以下の素晴らしいモデルとライブラリを使用しています：

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - オープンボキャブラリー物体検出モデル
- [Transformers](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) - Hugging Face Transformersライブラリ
- [LaMa](https://github.com/advimman/lama) - 大規模マスク画像修復モデル
- [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) - LaMaモデルの簡単な実装

これらの素晴らしいコードを共有していただき、心より感謝申し上げます。

## ライセンス

[ライセンス情報を記載]

## 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成
