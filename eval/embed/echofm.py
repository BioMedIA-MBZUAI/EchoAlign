import sys
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from echoalign.loader import file_kind, read_frames
from .echojepa import EchoJEPAEmbedder

REPO = Path(__file__).resolve().parents[2] / "models" / "EchoFM"
WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "echofm_vitl.pth"
CLIP_LEN, SIZE = 32, 224
MODEL_ARGS = ("num_frames", "t_patch_size", "pred_t_dim", "sep_pos_embed", "cls_embed", "norm_pix_loss")


def load_encoder(weights=WEIGHTS, repo_dir=REPO):
    repo = str(Path(repo_dir).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from EchoFM import models_mae

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = models_mae.mae_vit_large_patch16(**{k: ckpt["model_args"][k] for k in MODEL_ARGS})
    model.load_state_dict(ckpt["model"], strict=True)
    return model


class EchoFMEmbedder:
    name = "echofm"

    def __init__(self, weights=WEIGHTS, repo_dir=REPO, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_encoder(weights, repo_dir).to(self.device).eval()

    def preprocess(self, path):
        dicom = file_kind(Path(path)) == "dicom"
        frames = read_frames(path, n_frames=None, crop_to_region=dicom, pad_to_square=dicom)
        frames = [frames[i % len(frames)] for i in range(CLIP_LEN)]
        out = [TF.to_tensor(TF.resize(f, [SIZE, SIZE], interpolation=InterpolationMode.BILINEAR))
               for f in frames]
        return torch.stack(out, 1)

    @torch.inference_mode()
    def encode(self, batch):
        tokens, _, _ = self.model.forward_encoder(batch.to(self.device), mask_ratio=0.0)
        return dict(features=tokens.mean(1).float().cpu().numpy())

    study_level = EchoJEPAEmbedder.study_level
