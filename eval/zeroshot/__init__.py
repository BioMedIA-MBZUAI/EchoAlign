METHODS = ("echoalign", "echoprime", "panecho", "echoclip", "echoprime_prompts", "echoclip_prompts")


def build(name, **kwargs):
    if name == "echoalign":
        from .echoalign import EchoAlignPrompts
        return EchoAlignPrompts(**kwargs)
    if name == "echoprime":
        from .echoprime import EchoPrimeRetrieval
        return EchoPrimeRetrieval(**kwargs)
    if name == "panecho":
        from .panecho import PanEchoHeads
        return PanEchoHeads(**kwargs)
    if name == "echoclip":
        from .echoclip import EchoCLIPPrompts
        return EchoCLIPPrompts(**kwargs)
    if name == "echoprime_prompts":
        from .baseline_prompts import EchoPrimePrompts
        return EchoPrimePrompts(**kwargs)
    if name == "echoclip_prompts":
        from .baseline_prompts import EchoCLIPPromptsAnyTask
        return EchoCLIPPromptsAnyTask(**kwargs)
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
