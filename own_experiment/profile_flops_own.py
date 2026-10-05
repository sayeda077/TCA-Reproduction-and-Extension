import torch
import clip
from fvcore.nn import FlopCountAnalysis

class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual

    def forward(self, x):
        out = self.visual(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"

methods = [
    "EViT-0.0",    # CLIP baseline: no pruning
    "EViT-0.1",
    "ToME-0.1",
    "EViT-0.3",
    "EViT-0.5",
    "Ours-0.035",
    "Ours-0.105",
    "Ours-0.175",
]

for method in methods:
    model, _ = clip.load("ViT-B/16", method, device=device, jit=False)
    model.eval()

    encoder = ImageEncoder(model).to(device)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=model.dtype)

    analysis = FlopCountAnalysis(encoder, x)
    gflops = analysis.total() / 1e9

    print(f"{method:12s}: {gflops:.3f} GFLOPs")
    print("Unsupported ops:", analysis.unsupported_ops())
    print()

    del model, encoder, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
