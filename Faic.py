#!/usr/bin/env python3
import torch
import onnxruntime as rt
torch.set_grad_enabled(False)

from faic import Coordinator
if __name__ == "__main__":
    Coordinator.run()