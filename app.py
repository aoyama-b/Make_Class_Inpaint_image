import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw
import os

# モデルの初期化
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()
simple_lama = SimpleLama(device=device)

def make_mask(image, target_class, threshold):
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

    binary_mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(binary_mask)
    for box, label in zip(results["boxes"], results["text_labels"]):
        if label == target_class:
            x1, y1, x2, y2 = [round(i.item()) for i in box]
            draw.rectangle((x1, y1, x2, y2), fill=255)
    return binary_mask

def inpaint_lama(image, mask):
    result = simple_lama(image, mask)
    return result

def process_image(image, target_class, threshold):
    # マスク生成
    mask = make_mask(image, target_class, threshold)
    # インペイント実行
    inpainted = inpaint_lama(image, mask)
    return mask, inpainted

# Gradioインターフェースの作成
with gr.Blocks() as demo:
    gr.Markdown("# 画像インペイント処理")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="入力画像")
            target_class = gr.Textbox(label="対象クラス名", value="car")
            threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.2, label="検出閾値")
            process_btn = gr.Button("処理開始")
        
        with gr.Column():
            mask_output = gr.Image(label="生成されたマスク")
            output_image = gr.Image(label="インペイント結果")
    
    process_btn.click(
        fn=process_image,
        inputs=[input_image, target_class, threshold],
        outputs=[mask_output, output_image]
    )

if __name__ == "__main__":
    demo.launch() 