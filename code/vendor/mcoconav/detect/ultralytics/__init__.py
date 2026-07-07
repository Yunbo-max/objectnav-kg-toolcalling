"""Small YOLOv10-compatible stub used when upstream detection code is absent."""

import torch


class _Boxes:
    def __init__(self):
        self.cls = torch.empty(0, dtype=torch.long)
        self.conf = torch.empty(0)


class _Result:
    def __init__(self):
        self.names = {}
        self.boxes = _Boxes()


class YOLOv10:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, *args, **kwargs):
        return [_Result()]
