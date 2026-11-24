METHODS = ("echoalign", "echoprime", "panecho", "echoclip", "echojepa", "echofm")


def build(name, **kwargs):
    if name == "echoalign":
        from .echoalign import EchoAlignEmbedder
        return EchoAlignEmbedder(**kwargs)
    if name == "echoprime":
        from .echoprime import EchoPrimeEmbedder
        return EchoPrimeEmbedder(**kwargs)
    if name == "panecho":
        from .panecho import PanEchoEmbedder
        return PanEchoEmbedder(**kwargs)
    if name == "echoclip":
        from .echoclip import EchoCLIPEmbedder
        return EchoCLIPEmbedder(**kwargs)
    if name == "echojepa":
        from .echojepa import EchoJEPAEmbedder
        return EchoJEPAEmbedder(**kwargs)
    if name == "echofm":
        from .echofm import EchoFMEmbedder
        return EchoFMEmbedder(**kwargs)
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
