# End-to-End Inference (Photo/Sketch â†’ SR â†’ Quantum IGA Classifier)

This folder keeps an isolated inference pipeline and Hugging Face UI. Training stays in the module folders; drop trained artifacts into `EndToEnd/models/` (or point a config at them).

## What's here
- `pipeline.py` â€“ CLI to (optionally) convert a photo to a sketch with Pix2Pix, super-resolve with SRGAN, and classify IGA with the hybrid quantum model.
- `gradio_app.py` â€“ Gradio UI (HF Spaces-ready) with a Sketch/Photo toggle.
- `config.example.yaml` â€“ override model paths/labels.
- `requirements.txt` â€“ runtime deps for pipeline/UI.

## Expected model assets (put in `EndToEnd/models/`)
- `srgan_g.npz` â€“ SRGAN generator weights (copy from `SuperResolutionModule/models/g.npz` after training).
- `quantum_classifier.pt` â€“ `HybridNet` state_dict saved after training the quantum model.
- `quantum_scaler.pkl` *(optional)* â€“ fitted `StandardScaler` used before training.
- `quantum_pca.pkl` *(optional)* â€“ fitted PCA used before scaling.
- `pix2pix_generator.pth` *(optional, required for photo input)* â€“ Pix2Pix generator weights, e.g., `SketchModule/code/checkpoints/<exp>/latest_net_G.pth`. If present and set in the config, the pipeline can start from photos (`--from-photo` or the Gradio toggle).

Formats/locations by module:
- Pix2Pix (SketchModule): checkpoints under `SketchModule/code/checkpoints/<experiment>/` (`latest_net_G.pth`, `latest_net_D.pth`). Generated sketches live in `SketchModule/code/results/<experiment>/`. The end-to-end pipeline can either take these sketches directly or load `latest_net_G.pth` if you set `pix2pix_generator` in the config.
- SRGAN: generator weights in `SuperResolutionModule/models/g.npz` (TensorLayerX NPZ dict). Copy to `EndToEnd/models/srgan_g.npz`.
- Quantum classifier: save `torch.save(model.state_dict(), "quantum_classifier.pt")` after training `HybridNet` (PyTorch state_dict). Optional scaler/PCA via joblib (`.pkl`). Move to `EndToEnd/models/`.

If you change names/locations, mirror them in a config YAML and pass `--config` (CLI) or set `PIPELINE_CONFIG` (Gradio/Spaces).

## Usage (CLI)
```bash
cd EndToEnd
python -m pip install -r requirements.txt

# starting from a sketch
python pipeline.py --input ../path/to/your_sketch.png --output-dir ./output

# starting from a photo (requires pix2pix generator in config)
python pipeline.py --input ../path/to/your_photo.jpg --from-photo --output-dir ./output

# with a custom config
python pipeline.py --input ../path/to/your_sketch.png --config ./config.example.yaml
```
Outputs:
- If `--from-photo`: intermediate sketch at `<output-dir>/<name>_sketch.png`
- Super-resolved sketch: `<output-dir>/<name>_sr.png`
- Prediction JSON: `<output-dir>/<name>_prediction.json`

Use `--skip-superres` if your sketch is already high-res (classifier still runs).

## Converting photos to sketches (Pix2Pix)
1) Put photos in `SketchModule/data/dataset/photos/`.  
2) Run `SketchModule/code/test.py` with your trained Pix2Pix weights; outputs land in `SketchModule/code/results/<experiment>/`.  
3) Either feed those sketches to `EndToEnd/pipeline.py`, or set `pix2pix_generator` in the config to `latest_net_G.pth` and use `--from-photo` to let the pipeline handle it.

## Hugging Face Spaces
1) Space type: “Gradio (Python)”.  
2) Add model assets to `EndToEnd/models/`:
   - `srgan_g.npz` (from `SuperResolutionModule/models/g.npz`)
   - `quantum_classifier.pt`
   - Optional: `quantum_scaler.pkl`, `quantum_pca.pkl`
   - Optional (for photo input): Pix2Pix `latest_net_G.pth` and set it in the config
3) Start command (Settings → Runtime → Start command):
   ```bash
   pip install -r EndToEnd/requirements.txt && python EndToEnd/gradio_app.py
   ```
4) Env vars (Settings → Variables), if needed:
   - `PIPELINE_CONFIG`: path to a YAML config in the repo (e.g., `EndToEnd/config.prod.yaml`)
   - `PIPELINE_DEVICE`: `cpu` (default) or `cuda:0` if the Space has a GPU
5) UI behavior: upload Sketch or Photo (toggle). If photo + Pix2Pix weights are present, it converts; then SRGAN → quantum classifier; returns sketch, SR image, class probabilities, and predicted IGA label.
6) Tips: use Git LFS if model files are large; CPU Spaces work but slower; without Pix2Pix weights, photo mode is unavailable—use sketch input or add `pix2pix_generator` in the config.

## Training Playbook (per module)
- Pix2Pix sketches (`SketchModule`)
  1) Photos in `SketchModule/data/dataset/photos/`; ground-truth sketches in `SketchModule/data/dataset/sketches/`.
  2) Build pairs: `cd SketchModule/code && python combine_A_and_B.py --fold_A ../data/dataset/photos --fold_B ../data/dataset/sketches --fold_AB ../data/dataset/face2sketch`.
  3) Train: `python train.py --dataroot ../data/dataset --name results --model pix2pix --direction AtoB`.
  4) Generate sketches: `python test.py --dataroot ../data/dataset --name results --model pix2pix --direction AtoB --num_test <N>`.
  5) For end-to-end photo input, set `pix2pix_generator` in the config to `SketchModule/code/checkpoints/<experiment>/latest_net_G.pth`.

- SRGAN super-res (`SuperResolutionModule`)
  1) HR sketches in `SuperResolutionModule/data/HR/`; matching LR in `SuperResolutionModule/data/LR/` (same basenames).  
  2) Train: `cd SuperResolutionModule && python train.py --mode train`.  
  3) Copy `SuperResolutionModule/models/g.npz` to `EndToEnd/models/srgan_g.npz`.

- Quantum classifier (`QuantumModule`)
  1) Super-resolved sketches into `QuantumModule/helpers/superresolved_sketches/`; add `labels.csv` with `filename,label`.  
  2) Extract features: `cd QuantumModule/helpers && python feature_extractor.py` (writes `features.csv`; save PCA/scaler if you fit them).  
  3) Train hybrid model: `python ../quantum_module.py` (qiskit + torch installed). After training, save: `torch.save(model.state_dict(), "quantum_classifier.pt")`.  
  4) Move `quantum_classifier.pt` (and optional scaler/PCA `.pkl`) into `EndToEnd/models/`.

- Wire-up for inference
  - Ensure `EndToEnd/models/` contains `srgan_g.npz`, `quantum_classifier.pt`, optional `quantum_scaler.pkl`/`quantum_pca.pkl`, and optionally `latest_net_G.pth` if you want `--from-photo`. Update the config if paths differ.
