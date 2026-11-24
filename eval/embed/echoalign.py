import numpy as np
import torch

from echoalign.loader import load_echo
from echoalign.pooling import pool_studies


class EchoAlignEmbedder:
    name = "echoalign"

    def __init__(self, weights, device=None):
        from echoalign.model import EchoAlign
        self.model = EchoAlign(weights, device=device)

    def preprocess(self, path):
        return load_echo(path)

    def encode(self, batch):
        g, c = self.model.encode_clips(batch)
        return dict(global_vector=np.asarray(g, np.float32), structures=np.asarray(c, np.float32))

    def study_level(self, clips, study_id):
        uniq, group = np.unique(study_id, return_inverse=True)
        with torch.inference_mode():
            vectors = pool_studies(self.model.pooler, clips["global_vector"], clips["structures"],
                                   group_index=group.tolist())
        return dict(study_id=uniq, vectors=np.asarray(vectors, np.float32))
