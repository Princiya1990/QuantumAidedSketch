"""
End-to-end inference pipeline that:
1) Takes a sketch image.
2) Super-resolves it with the trained SRGAN generator.
3) Extracts CNN features.
4) Classifies the image with the trained quantum-enhanced model to predict IGA acne class.

Training is intentionally kept separate. Place trained weights and scalers inside the models/ folder
or point to them via a config file/CLI flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

# Ensure TensorLayerX uses the same backend as training
os.environ.setdefault("TL_BACKEND", "tensorflow")

import numpy as np
import tensorlayerx as tlx
from PIL import Image
from joblib import load as joblib_load
import torch
from torch import nn
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.connectors import TorchConnector
import yaml

# Local imports: keep this folder isolated from the rest of the repo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "SuperResolutionModule") not in sys.path:
    sys.path.append(str(ROOT / "SuperResolutionModule"))
from srgan import SRGAN_g  # type: ignore  # imported via sys.path hack

if str(ROOT / "SketchModule" / "code") not in sys.path:
    sys.path.append(str(ROOT / "SketchModule" / "code"))
try:
    from models import networks  # type: ignore
except Exception:
    networks = None

DEFAULT_LABELS = ["IGA-0", "IGA-1", "IGA-2", "IGA-3", "IGA-4"]


@dataclass
class PipelineConfig:
    model_dir: Path
    sr_generator: Path
    quantum_classifier: Path
    scaler: Optional[Path] = None
    pca: Optional[Path] = None
    label_names: List[str] = field(default_factory=lambda: DEFAULT_LABELS)
    pix2pix_generator: Optional[Path] = None


def _resolve_path(base: Path, candidate: Union[str, Path]) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return base / candidate_path


def load_config(config_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Load YAML config if provided; otherwise fall back to defaults under models/."""
    here = Path(__file__).resolve().parent
    default_model_dir = here / "models"
    default_sr = default_model_dir / "srgan_g.npz"
    default_classifier = default_model_dir / "quantum_classifier.pt"
    default_scaler = default_model_dir / "quantum_scaler.pkl"
    default_pca = default_model_dir / "quantum_pca.pkl"
    default_pix2pix = ROOT / "SketchModule" / "code" / "checkpoints" / "results" / "latest_net_G.pth"

    cfg_dict = {}
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
        base_for_relative = config_path.parent
    else:
        base_for_relative = here

    model_dir = _resolve_path(base_for_relative, cfg_dict.get("model_dir", default_model_dir))
    sr_generator = _resolve_path(base_for_relative, cfg_dict.get("sr_generator", default_sr))
    quantum_classifier = _resolve_path(base_for_relative, cfg_dict.get("quantum_classifier", default_classifier))
    scaler = cfg_dict.get("scaler")
    pca = cfg_dict.get("pca")

    scaler_path = _resolve_path(base_for_relative, scaler) if scaler else default_scaler
    pca_path = _resolve_path(base_for_relative, pca) if pca else default_pca
    label_names = cfg_dict.get("label_names", DEFAULT_LABELS)
    pix2pix_path = cfg_dict.get("pix2pix_generator")
    pix2pix_resolved = _resolve_path(base_for_relative, pix2pix_path) if pix2pix_path else default_pix2pix

    return PipelineConfig(
        model_dir=model_dir,
        sr_generator=sr_generator,
        quantum_classifier=quantum_classifier,
        scaler=scaler_path if scaler_path.exists() else None,
        pca=pca_path if pca_path.exists() else None,
        label_names=label_names,
        pix2pix_generator=pix2pix_resolved if pix2pix_resolved.exists() else None,
    )


def _ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    return path


def _pad_or_trim(vec: np.ndarray, target: int) -> np.ndarray:
    if len(vec) >= target:
        return vec[:target].astype(np.float32)
    return np.pad(vec, (0, target - len(vec)), mode="constant").astype(np.float32)


class SuperResolutionEngine:
    """Lightweight wrapper to load SRGAN generator weights and run inference."""

    def __init__(self, weights_path: Path):
        self.weights_path = _ensure_exists(weights_path, "SRGAN generator weights")
        self.generator = SRGAN_g()
        # match training build shape; generator supports arbitrary H/W at inference
        self.generator.init_build(tlx.nn.Input(shape=(1, 3, 96, 96)))
        self.generator.load_weights(str(self.weights_path), format="npz_dict")
        self.generator.set_eval()
        tlx.set_device("CPU")

    @staticmethod
    def _to_tensor(image: Image.Image) -> tlx.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = arr / 127.5 - 1.0  # map to [-1, 1] to match training
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return tlx.ops.convert_to_tensor(arr)

    @staticmethod
    def _to_image(tensor: tlx.Tensor) -> Image.Image:
        arr = tlx.ops.convert_to_numpy(tensor)
        arr = np.asarray((arr + 1.0) * 127.5, dtype=np.uint8)
        arr = np.transpose(arr[0], (1, 2, 0))
        return Image.fromarray(arr)

    def enhance(self, image: Image.Image) -> Image.Image:
        lr_tensor = self._to_tensor(image)
        sr_tensor = self.generator(lr_tensor)
        return self._to_image(sr_tensor)


class SketchFeatureExtractor:
    """ResNet18 trunk used as a fixed feature extractor (same as training script)."""

    def __init__(self, device: str = "cpu"):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __call__(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(tensor).cpu().numpy().flatten()
        return feats


class Pix2PixGenerator:
    """Lightweight loader for the trained Pix2Pix generator (netG)."""

    def __init__(
        self,
        weights_path: Path,
        device: str = "cpu",
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        netG: str = "unet_256",
        norm: str = "batch",
        use_dropout: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ):
        if networks is None:
            raise ImportError("Could not import SketchModule/code/models/networks.py. Check sys.path and repo structure.")
        self.device = device
        self.netG = networks.define_G(
            input_nc,
            output_nc,
            ngf,
            netG,
            norm,
            use_dropout,
            init_type,
            init_gain,
            gpu_ids=[],
        ).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        self.netG.load_state_dict(state_dict)
        self.netG.eval()
        # Match pix2pix test preprocessing: resize to load_size (256) and normalize to [-1, 1]
        self.transform = T.Compose(
            [
                T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        fake = self.netG(x)[0]
        fake = (fake.clamp(-1, 1) + 1) * 0.5  # back to [0,1]
        fake = fake.mul(255).byte().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(fake)


class HybridNet(nn.Module):
    """Hybrid quantum-classical head defined in the training code."""

    def __init__(self, quantum_model: TorchConnector, n_classes: int = 5):
        super().__init__()
        self.q_layer = quantum_model
        self.fc = nn.Linear(quantum_model.output_shape[0], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.q_layer(x)
        out = self.fc(out)
        return out


class QuantumAcneClassifier:
    """Loads the trained TorchConnector + linear head and runs prediction."""

    def __init__(
        self,
        weights_path: Path,
        scaler_path: Optional[Path],
        pca_path: Optional[Path],
        num_qubits: int = 10,
        label_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.weights_path = _ensure_exists(weights_path, "Quantum classifier weights")
        self.num_qubits = num_qubits
        self.device = device
        self.labels = label_names or DEFAULT_LABELS

        backend = Aer.get_backend("aer_simulator_statevector")
        quantum_instance = QuantumInstance(backend)

        ansatz = RealAmplitudes(num_qubits, reps=2)
        vqc = VQC(feature_map=ansatz, ansatz=ansatz, optimizer=None, quantum_instance=quantum_instance)
        quantum_layer = TorchConnector(vqc)

        self.model = HybridNet(quantum_layer, n_classes=len(self.labels))
        self.model.load_state_dict(torch.load(self.weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.scaler = joblib_load(scaler_path) if scaler_path and Path(scaler_path).exists() else None
        self.pca = joblib_load(pca_path) if pca_path and Path(pca_path).exists() else None

    def _prepare_features(self, features: np.ndarray) -> torch.Tensor:
        vec = features.astype(np.float32)
        if self.pca is not None:
            vec = self.pca.transform([vec])[0]
        if self.scaler is not None:
            vec = self.scaler.transform([vec])[0]
        vec = _pad_or_trim(vec, self.num_qubits)
        return torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict(self, features: np.ndarray) -> dict:
        x = self._prepare_features(features)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx] if self.labels else idx,
            "raw_index": idx,
            "probabilities": probs,
        }


class AcnePipeline:
    """End-to-end runner that wires SR -> feature extraction -> quantum classifier."""

    def __init__(self, config: PipelineConfig, device: str = "cpu"):
        self.config = config
        self.superres = SuperResolutionEngine(config.sr_generator)
        self.extractor = SketchFeatureExtractor(device=device)
        self.classifier = QuantumAcneClassifier(
            weights_path=config.quantum_classifier,
            scaler_path=config.scaler,
            pca_path=config.pca,
            num_qubits=10,
            label_names=config.label_names,
            device=device,
        )
        self.pix2pix = (
            Pix2PixGenerator(config.pix2pix_generator, device=device) if config.pix2pix_generator else None
        )

    @staticmethod
    def _load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input
        return Image.open(image_input)

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        skip_superres: bool = False,
        from_photo: bool = False,
    ) -> dict:
        source_image = self._load_image(image_input)
        if from_photo:
            if not self.pix2pix:
                raise ValueError("from_photo=True but no Pix2Pix generator configured.")
            sketch_image = self.pix2pix(source_image)
        else:
            sketch_image = source_image

        sr_image = sketch_image if skip_superres else self.superres.enhance(sketch_image)
        features = self.extractor(sr_image)
        pred = self.classifier.predict(features)
        probs_dict = {
            label: float(prob) for label, prob in zip(self.classifier.labels, pred["probabilities"])
        }
        return {
            "prediction": pred["label"],
            "probabilities": probs_dict,
            "sketch_image": sketch_image,
            "superresolved_image": sr_image,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sketch -> SR -> quantum acne classification.")
    parser.add_argument("--input", required=True, help="Path to a sketch image.")
    parser.add_argument(
        "--output-dir",
        default="EndToEnd/output",
        help="Where to save the super-resolved image and prediction JSON.",
    )
    parser.add_argument("--config", help="Optional YAML config overriding model paths.")
    parser.add_argument(
        "--skip-superres",
        action="store_true",
        help="Bypass SR if you already have a high-res sketch.",
    )
    parser.add_argument(
        "--from-photo",
        action="store_true",
        help="If set, treats the input as a photo and first runs Pix2Pix (requires pix2pix_generator in config).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the feature extractor and classifier (e.g., cpu or cuda:0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = AcnePipeline(cfg, device=args.device)

    result = pipeline.run(args.input, skip_superres=args.skip_superres, from_photo=args.from_photo)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sketch_out_path = output_dir / (Path(args.input).stem + "_sketch.png")
    result["sketch_image"].save(sketch_out_path)

    image_path = output_dir / (Path(args.input).stem + "_sr.png")
    result["superresolved_image"].save(image_path)

    report = {
        "input": str(Path(args.input).resolve()),
        "sketch_image": str(sketch_out_path.resolve()),
        "superresolved_image": str(image_path.resolve()),
        "prediction": result["prediction"],
        "probabilities": result["probabilities"],
    }
    report_path = output_dir / (Path(args.input).stem + "_prediction.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Pipeline] Super-resolved sketch saved to: {image_path}")
    print(f"[Pipeline] Prediction JSON saved to: {report_path}")


if __name__ == "__main__":
    main()
