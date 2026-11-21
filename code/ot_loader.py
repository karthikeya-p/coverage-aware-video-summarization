"""Load OpenTAD's AdaTAD (VideoMAE-L + adapters + ActionFormer head) without installing mmaction2 / full mmcv.

OpenTAD imports mmaction2 at package import time for backbones we never use. We register lightweight placeholder
modules for the `mmaction` namespace (and for the compiled NMS extension when absent), then import the real
OpenTAD model code. Only `get_sinusoid_encoding` needs a real implementation; the positional table itself is
loaded from the checkpoint.
"""
import importlib.abc, importlib.machinery, os, sys, types
import torch


def _sinusoid(n_position, embed_dims):
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = torch.pow(10000, -((vec - vec % 2) / embed_dims)).view(1, -1)
    table = torch.arange(n_position).view(-1, 1) * vec
    table[:, 0::2].sin_(); table[:, 1::2].cos_()
    return table.to(torch.float32).unsqueeze(0)


class _Placeholder(type):
    """Any attribute of a placeholder module is a class that can be subclassed, called, or used as a decorator."""
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make(name)

    def __call__(cls, *a, **k):
        if len(a) == 1 and not k and (isinstance(a[0], type) or callable(a[0])):
            return a[0]                      # used as a decorator: @X or @X.register_module()(cls)
        return cls                           # used as a factory: X.register_module() -> decorator


def _make(name):
    return _Placeholder(name, (), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "get_sinusoid_encoding":
            return _sinusoid
        return _make(name)


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, roots):
        self.roots = tuple(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []; return m

    def exec_module(self, module):
        pass


def setup(opentad_dir):
    roots = []
    for name in ("mmaction", "nms_1d_cpu", "Align1D", "boundary_max_pooling_cuda"):
        try:
            if name == "mmaction":
                raise ImportError          # never use the real one: its import chain needs full mmcv
            __import__(name)
        except Exception:
            roots.append(name)
    for extra in ("imgaug", "pytorchvideo", "decord", "wandb", "torchvision"):
        try:
            __import__(extra)
        except Exception:
            roots.append(extra)
    sys.meta_path.insert(0, _StubFinder(roots))
    if opentad_dir not in sys.path:
        sys.path.insert(0, opentad_dir)
    return roots
