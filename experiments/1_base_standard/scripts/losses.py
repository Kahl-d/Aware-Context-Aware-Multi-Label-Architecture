import torch
import torch.nn as nn
import logging
from config import NUM_THEMES
logger=logging.getLogger(__name__)
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self,theme_weights=None):
        super().__init__()
        self.bce=nn.BCEWithLogitsLoss(reduction="none")
        if theme_weights is not None: self.register_buffer("theme_weights",theme_weights)
        else: self.theme_weights=None
    def forward(self,logits,targets,mask=None):
        loss=self.bce(logits,targets)
        if self.theme_weights is not None: loss=loss*self.theme_weights.view(1,1,-1)
        if mask is not None:
            m=mask.unsqueeze(-1).float();loss=loss*m;return loss.sum()/m.sum().clamp(min=1)
        return loss.mean()
def build_loss_from_config(config,theme_weights_list=None):
    tw=None
    if config.loss.theme_weights is not None:
        w=config.loss.theme_weights
        if isinstance(w,(list,tuple)) and len(w)==NUM_THEMES: tw=torch.tensor(w,dtype=torch.float32)
    elif theme_weights_list is not None: tw=torch.tensor(theme_weights_list,dtype=torch.float32)
    return WeightedBCEWithLogitsLoss(theme_weights=tw)
