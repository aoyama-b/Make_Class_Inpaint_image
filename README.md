# 画像インペイント処理アプリケーション

このアプリケーションは、指定されたオブジェクトを画像から自動的に検出し、その部分をインペイント（修復）するツールです。Grounding DINOとLaMaモデルを使用して、高品質な画像修復を実現します。

## 機能

- 画像内の特定のオブジェクト（車、人物など）の自動検出
- 検出されたオブジェクトのマスク生成
- LaMaモデルを使用した高品質なインペイント処理
- Gradioを使用した直感的なWebインターフェース

## 必要条件

- Docker
- Docker Compose
- NVIDIA GPU（推奨）とNVIDIA Container Toolkit

## インストール方法

1. リポジトリのクローン:
```bash
git clone https://github.com/aoyama-b/Make_Class_Inpaint_image.git
cd https://github.com/aoyama-b/Make_Class_Inpaint_image.git
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

## 使用方法

1. アプリケーションの起動:
```bash
docker-compose up --build
```

2. ブラウザで以下のURLにアクセス:
```
http://localhost:7860
```

3. 使用方法:
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
- Docker

## ライセンス

[ライセンス情報を記載]

## 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成
