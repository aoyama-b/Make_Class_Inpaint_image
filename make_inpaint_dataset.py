import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw
import numpy as np

def inpaint_lama(input_img_path, mask):
    image = Image.open(input_img_path)
    result = simple_lama(image, mask)
    return result

def make_mask(input_img_path, target_class, threshold, model, processor, device):
    image = Image.open(input_img_path).convert("RGB")
    width, height = image.size
    text = target_class + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=[(height, width)],
        box_threshold=threshold,
        text_threshold=threshold
    )[0]

    #バイナリマスク生成
    binary_mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(binary_mask)
    for box, label in zip(results["boxes"], results["text_labels"]):
        if label == target_class:
            x1, y1, x2, y2 = [round(i.item()) for i in box]
            draw.rectangle((x1, y1, x2, y2), fill=255)
    return binary_mask

def main():
    parser = argparse.ArgumentParser(
        description="Batch inpainting of a specified class using Grounding DINO + LaMa."
    )
    parser.add_argument(
        "--input_folder", "-i",
        required=True,
        help="入力画像を格納したフォルダのパス"
    )
    parser.add_argument(
        "--output_folder", "-o",
        required=True,
        help="出力画像を保存するフォルダのパス"
    )
    parser.add_argument(
        "--target_class", "-c",
        default="car.",
        help="マスクを生成する対象クラス名（例: car, person, dog）"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.2,
        help="Grounding DINO のボックス検出・テキスト検出閾値"
    )
    args = parser.parse_args()

    # 出力フォルダ作成
    os.makedirs(args.output_folder, exist_ok=True)

    #初期化
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    simple_lama = SimpleLama()

    # バッチ処理
    for fname in os.listdir(args.input_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(args.input_folder, fname)
        # マスク生成
        mask = make_mask(input_path, args.target_class, args.threshold, model, processor, device)
        # インペイント実行
        inpainted = inpaint_lama(input_path, mask, simple_lama)
        # 保存
        output_path = os.path.join(args.output_folder, fname)
        inpainted.save(output_path)
        print(f"[Saved] {output_path}")

if __name__ == "__main__":
    main()