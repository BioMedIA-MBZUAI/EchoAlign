import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from echoalign.loader import file_kind, read_frames

REPO = Path(__file__).resolve().parents[2] / "models" / "EchoJEPA"
WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "vitl-vmix22m-pt220-c55.pt"
CLIP_LEN, FPS, SHORT_SIDE, CROP = 16, 8, 256, 224
NPZ_FPS = 24
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_encoder(weights=WEIGHTS, repo_dir=REPO):
    repo = str(Path(repo_dir).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import src.models.vision_transformer as vit

    model = vit.vit_large(img_size=CROP, num_frames=CLIP_LEN, patch_size=16, tubelet_size=2,
                          uniform_power=True, use_rope=True)
    state = torch.load(weights, map_location="cpu", weights_only=False)["target_encoder"]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


class EchoJEPAEmbedder:
    name = "echojepa"

    def __init__(self, weights=WEIGHTS, repo_dir=REPO, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_encoder(weights, repo_dir).to(self.device).eval()

    def preprocess(self, path):
        if str(path).endswith(".npz"):
            frames = read_frames(path, n_frames=CLIP_LEN, stride=NPZ_FPS // FPS, crop_to_region=False,
                                 pad_to_square=False)
        else:
            dicom = file_kind(Path(path)) == "dicom"
            frames = read_frames(path, n_frames=CLIP_LEN, target_fps=FPS, crop_to_region=dicom, pad_to_square=dicom)
        out = []
        for f in frames:
            f = TF.resize(f, SHORT_SIDE, interpolation=InterpolationMode.BILINEAR)
            out.append(TF.to_tensor(TF.center_crop(f, [CROP, CROP])))
        x = TF.normalize(torch.stack(out, 0), MEAN, STD)
        return x.permute(1, 0, 2, 3).contiguous()

    @torch.inference_mode()
    def encode(self, batch):
        tokens = self.model(batch.to(self.device))
        return dict(features=tokens.mean(1).float().cpu().numpy())

    def study_level(self, clips, study_id):
        uniq, group = np.unique(study_id, return_inverse=True)
        sums = np.zeros((len(uniq), clips["features"].shape[1]), np.float64)
        np.add.at(sums, group, clips["features"])
        return dict(study_id=uniq, features=(sums / np.bincount(group)[:, None]).astype(np.float32))
