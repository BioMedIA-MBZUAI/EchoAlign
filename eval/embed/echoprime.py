from pathlib import Path

import numpy as np
import torch
import torchvision

from echoalign.loader import file_kind, read_frames
from eval.embed.crop import crop_and_scale
from eval.zeroshot.echoprime import import_echoprime

STILLS = ("skip", "repeat")


class EchoPrimeEmbedder:
    name = "echoprime"

    def __init__(self, repo_dir, device=None, encoder_path=None, stills="skip"):
        if stills not in STILLS:
            raise ValueError(f"unknown stills mode {stills!r}; choose from {STILLS}")
        self.stills = stills
        repo, ep_utils, EchoPrime = import_echoprime(repo_dir)
        self.utils = ep_utils
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ep = EchoPrime.__new__(EchoPrime)
        # The same models and settings as EchoPrime.__init__.
        encoder = torchvision.models.video.mvit_v2_s()
        encoder.head[-1] = torch.nn.Linear(encoder.head[-1].in_features, 512)
        self.encoder_path = Path(encoder_path) if encoder_path else repo / "model_data/weights/echo_prime_encoder.pt"
        encoder.load_state_dict(torch.load(self.encoder_path, map_location=self.device))
        views = torchvision.models.convnext_base()
        views.classifier[-1] = torch.nn.Linear(views.classifier[-1].in_features, 11)
        views.load_state_dict(torch.load(repo / "model_data/weights/view_classifier.pt", map_location=self.device))
        ep.echo_encoder = encoder.to(self.device).eval()
        ep.view_classifier = views.to(self.device).eval()
        ep.frames_to_take, ep.frame_stride, ep.video_size = 32, 2, 224
        ep.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        ep.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        ep.device = self.device
        self.ep = ep

    def _finish(self, pixels):
        ep = self.ep
        pixels = pixels[:ep.frames_to_take]
        x = np.zeros((len(pixels), 224, 224, 3))
        for i in range(len(x)):
            x[i] = crop_and_scale(pixels[i])
        x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
        x.sub_(ep.mean).div_(ep.std)
        if x.shape[1] < ep.frames_to_take:
            pad = torch.zeros((3, ep.frames_to_take - x.shape[1], ep.video_size, ep.video_size), dtype=torch.float)
            x = torch.cat((x, pad), dim=1)
        return x[:, 0:ep.frames_to_take:ep.frame_stride, :, :]

    def preprocess(self, path):
        if file_kind(Path(path)) == "dicom":
            import pydicom
            dcm = pydicom.dcmread(path)
            if int(pydicom.__version__.split(".")[0]) >= 3:
                # EchoPrime's pydicom 2.3.1 keeps the stored color space (mostly YCbCr); mask_outside_ultrasound converts it.
                dcm.pixel_array_options(as_rgb=False)
            pixels = dcm.pixel_array
            if not (pixels.ndim < 3 or pixels.shape[2] == 3):
                pixels = self.utils.mask_outside_ultrasound(dcm.pixel_array)
                return self._finish(pixels)
            if self.stills == "skip":
                return None
            if pixels.dtype != np.uint8:
                raise ValueError(f"{path}: {pixels.dtype} still image; only 8-bit stills are supported")
            import cv2
            if pixels.ndim == 2:
                frame = np.repeat(pixels[..., None], 3, axis=2).astype(np.uint8)
            elif str(dcm.get("PhotometricInterpretation", "")).startswith("YBR"):
                frame = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_YUV2BGR)
            else:
                frame = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2BGR)
            return self._finish(np.stack([frame] * self.ep.frames_to_take))
        frames = read_frames(path, n_frames=None, crop_to_region=False, pad_to_square=False)
        if len(frames) == 1:
            if self.stills == "skip":
                return None
            frames = frames * self.ep.frames_to_take
        return self._finish(np.stack([np.asarray(f) for f in frames]))

    @torch.inference_mode()
    def encode(self, batch):
        feats = self.ep.embed_videos(batch)
        views = self.ep.get_views(batch).argmax(1)
        return dict(features=feats.float().cpu().numpy(), views=views.cpu().numpy().astype(np.int64))
