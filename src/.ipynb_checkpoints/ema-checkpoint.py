import torch

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.device = next(model.parameters()).device

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]
