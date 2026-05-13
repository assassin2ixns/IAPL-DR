from .clip_models import CLIPModel
from .dream_cs import DREAMCSModel

VALID_NAMES = [
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
]

def build_model(args):
    if args.backbone.startswith("CLIP:"):
        assert args.backbone in VALID_NAMES
        if getattr(args, 'method', 'iapl') == 'dream_cs':
            return DREAMCSModel(args)
        return CLIPModel(args)
        # return SeArModel(args)
