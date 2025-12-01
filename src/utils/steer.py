import torch

class SteeringHook:
    def __init__(self, model, layer_idx, vec, alpha):
        self.model = model
        self.layer_idx = layer_idx
        self.vec = torch.tensor(vec).to(model.device)
        self.alpha = alpha
        self._hook = None

    def __enter__(self):
        def hook(module, inp, out):
            h = out[0]  # hidden state
            h = h + self.alpha * self.vec
            return (h, out[1]) if isinstance(out, tuple) else h

        block = self.model.model.layers[self.layer_idx]
        self._hook = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hook.remove()
