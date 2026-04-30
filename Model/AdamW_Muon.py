"""
HybridOptimizer — AdamW + Muon combined optimiser.

Parameter routing:
  - 2-D+ weight matrices (excluding embeddings)     → Muon
  - Norms, biases, and 1-D parameters               → AdamW (no weight decay)
  - All other < 2-D parameters                       → AdamW (with weight decay)

Exposes the same `zero_grad`, `step`, and `state_dict` interface as a standard
PyTorch optimiser so it can be used as a drop-in replacement in the training loop.
"""

class HybridOptimizer:
    def __init__(self,model,lr=Config.lr,weight_decay=0.01):
        adamw_decay=[]
        adamw_nodecay=[]
        muon_params=[]

        for name,p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim>=2 and "embedding" not in name:
                muon_params.append(p)
            else:
                if p.ndim<2 or "norm" in name.lower() or "bias" in name.lower():
                    adamw_nodecay.append(p)
                else:
                    adamw_decay.append(p)

        self.adamw=torch.optim.AdamW(
            [
                {"params":adamw_decay,"weight_decay":weight_decay},
                {"params":adamw_nodecay,"weight_decay":0.0},
            ],
            lr=lr,
            fused=True
        )
        self.muon=torch.optim.Muon(
            muon_params,
            lr=lr,
            adjust_lr_fn="match_rms_adamw"
        )

    def zero_grad(self,set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adamw.step()
        self.muon.step()

    def state_dict(self):
        return {
            "adamw": self.adamw.state_dict(),
            "muon": self.muon.state_dict()
        }
