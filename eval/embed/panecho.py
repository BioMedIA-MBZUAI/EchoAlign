import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from echoalign.loader import read_frames
from eval.zeroshot.panecho import hub_load

CLIP_LEN = 16
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class PanEchoEmbedder:
    name = "panecho"

    def __init__(self, source="models/PanEcho", local=True, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = hub_load(source, local, backbone_only=True, clip_len=CLIP_LEN)
        self.model = model.to(self.device).eval()

    def preprocess(self, path):
        frames = read_frames(path, n_frames=CLIP_LEN, crop_to_region=False, pad_to_square=False)
        out = []
        for f in frames:
            x = TF.to_tensor(TF.resize(f, [224, 224], interpolation=InterpolationMode.BILINEAR))
            out.append(x[[2, 1, 0]])
        x = TF.normalize(torch.stack(out, 0), MEAN, STD)
        return x.permute(1, 0, 2, 3).contiguous()

    @torch.inference_mode()
    def encode(self, batch):
        return dict(features=self.model(batch.to(self.device)).float().cpu().numpy())
