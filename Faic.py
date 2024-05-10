#!/usr/bin/env python3
import torch

from faic import Coordinator

torch.set_grad_enabled(False)

if __name__ == "__main__":
    Coordinator.run()
