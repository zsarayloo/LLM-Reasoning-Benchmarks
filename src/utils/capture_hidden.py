# src/utils/capture_hidden.py

from __future__ import annotations
from typing import Dict, List, Any, Callable, Optional

import torch
from torch import Tensor


class HiddenCapture:
    """
    Utility to capture hidden states from specific layers of a HuggingFace-style
    transformer model using forward hooks.

    Usage:
        capturer = HiddenCapture(layer_names=["model.layers.15", "model.layers.20"])
        capturer.register(model)

        # Run model(...)
        # Then access capturer.hiddens["model.layers.15"], etc.

        capturer.remove()  # remove all hooks when done
    """

    def __init__(self, layer_names: List[str]):
        self.layer_names = set(layer_names)
        self.hiddens: Dict[str, Tensor] = {}
        self.handles: List[Any] = []

    def _make_hook(self, name: str) -> Callable:
        """
        Build a forward hook function for a given module name.
        """

        def hook(module, inputs, output):
            # Many transformer blocks output a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output

            # Detach & move to cpu to avoid keeping graph / GPU memory
            self.hiddens[name] = hs.detach().cpu()

        return hook

    def register(self, model: torch.nn.Module) -> None:
        """
        Register forward hooks on the given model for each layer in layer_names.
        You must know the correct module names (from model.named_modules()).
        """
        self.hiddens.clear()
        self.handles.clear()

        for name, module in model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

        if not self.handles:
            print(
                "[HiddenCapture] WARNING: No hooks were registered. "
                "Check your layer_names vs model.named_modules()."
            )

    def clear(self) -> None:
        """Clear stored hidden states but keep hooks."""
        self.hiddens.clear()

    def remove(self) -> None:
        """Remove all hooks and clear stored hidden states."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.hiddens.clear()


def last_token(t: Tensor) -> Tensor:
    """
    Return last-token embedding from a [batch, seq_len, hidden_dim] tensor.
    """
    if t.ndim != 3:
        raise ValueError(
            f"last_token expects a 3D tensor [batch, seq, hidden], got shape {tuple(t.shape)}"
        )
    return t[:, -1, :]
