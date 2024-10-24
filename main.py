#!/usr/bin/env python3
import torch

from faic import coordinator

torch.set_grad_enabled(False)

if __name__ == "__main__":
    coordinator.run()
