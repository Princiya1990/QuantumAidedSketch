"""
Gradio app for Hugging Face Spaces deployment.

Expected assets in EndToEnd/models/ (or override via PIPELINE_CONFIG):
- srgan_g.npz              : SRGAN generator weights
- quantum_classifier.pt    : HybridNet state_dict
- quantum_scaler.pkl       : Optional StandardScaler
- quantum_pca.pkl          : Optional PCA used before scaling
"""
from pathlib import Path
import os

import gradio as gr

from pipeline import AcnePipeline, load_config


def build_pipeline() -> AcnePipeline:
    config_path = os.environ.get("PIPELINE_CONFIG")
    cfg = load_config(config_path) if config_path else load_config()
    return AcnePipeline(cfg, device=os.environ.get("PIPELINE_DEVICE", "cpu"))


PIPELINE = build_pipeline()


def infer(image, input_type):
    from_photo = input_type == "Photo"
    result = PIPELINE.run(image, from_photo=from_photo)
    return (
        result["sketch_image"],
        result["superresolved_image"],
        result["probabilities"],
        result["prediction"],
    )


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Input image (sketch or photo)"),
        gr.Radio(["Sketch", "Photo"], value="Sketch", label="Input type"),
    ],
    outputs=[
        gr.Image(label="Generated/Provided sketch"),
        gr.Image(label="Super-resolved sketch"),
        gr.Label(label="Class probabilities"),
        gr.Textbox(label="Predicted IGA class"),
    ],
    title="Acne Sketch Super-Resolution + Quantum Recognition",
    description="Upload a sketch (or a photo if Pix2Pix weights are configured). The app converts (if needed), upscales with SRGAN, and classifies IGA severity with the hybrid quantum model.",
    allow_flagging="never",
    examples=None,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
