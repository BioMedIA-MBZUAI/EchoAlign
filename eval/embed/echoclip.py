import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from echoalign.loader import read_frames
from eval.embed.crop import crop_and_scale

MAX_FRAMES = 20


class EchoCLIPEmbedder:
    name = "echoclip"

    def __init__(self, model_id="hf-hub:mkaichristensen/echo-clip", device=None):
        import open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model_id, precision="fp32",
                                                                                    device=self.device)
        self.model.eval()

    def preprocess(self, path):
        frames = read_frames(path, n_frames=None, crop_to_region=False, pad_to_square=False)[0:40:2]
        out = []
        for f in frames:
            bgr = np.ascontiguousarray(np.asarray(f)[:, :, ::-1])
            out.append(self.preprocess_val(Image.fromarray(crop_and_scale(bgr, (224, 224)))))
        x = torch.zeros((MAX_FRAMES, *out[0].shape))
        x[:len(out)] = torch.stack(out)
        return x, len(out)

    @torch.inference_mode()
    def encode(self, batch):
        x, n = batch
        b, t = x.shape[:2]
        e = F.normalize(self.model.encode_image(x.flatten(0, 1).to(self.device)).float(), dim=-1).reshape(b, t, -1)
        mask = torch.arange(t)[None] < torch.as_tensor(n)[:, None]
        return dict(frame_features=(e.cpu() * mask[..., None]).numpy().astype(np.float32),
                    n_frames=np.asarray(n, np.int64))
