## Make_Class_Inpaint_image
このコードはオープンボキャブラリー物体検出器である[Groundig DINO](https://github.com/IDEA-Research/GroundingDINO)を用いて検出した画像内のオブジェクトを[LaMa](https://github.com/advimman/lama)を用いて削除するものです。

## Installation and Get Started

Required environments:
- Linux
- Python
- PyTorch
- CUDA(GPU利用可なら)

Install:

```
pip install simple-lama-inpainting
pip install transformers
```

## Usage:
- Inference:画像フォルダを用いた推論には以下のコードを用います。
コマンドラインでは入力画像ディレクトリパス、出力画像ディレクトリパス、検出したいオブジェクトクラス、Grounding DINOの検出に用いる閾値を設定することができます。
```bash
CUDA_VISIBLE_DEVICES={GPU ID} python make_inpaint_dataset.py \
-i {your input image folder path} \
-o {your output image folder path} \
-c {Object class to be detected} \
-t {Object detector threshold}
```

- Gradio:未実装

## Acknowledgments:
以下に実装に用いたモデル・モジュールのリンクを添付します。
素晴らしいコードを共有していただきありがとうございます。
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Transformers](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)
- [LaMa](https://github.com/advimman/lama)
- [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
