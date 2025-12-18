# prune_cvn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from typing import Optional, List, Tuple
import pprint
from models.Models import MixedCVNN
from pruning_utils import collect_structured_prunable_params, structured_prune_model, make_pruning_permanent, model_sparsity_report


if __name__ == "__main__":
    dummy_config = {
        "conv1":"binary","conv2":"binary","conv3":"binary","conv4":"binary",
        "conv5":"binary","conv6":"binary","conv7":"binary","conv8":"binary",
        "conv9":"binary","conv10":"binary",
        "fc1":"binary","fc2":"binary","final":"binary"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixedCVNN(model_config=dummy_config, image_channels=3, filter_dimension=3, num_classes=101)
    model.to(device)

    print("=== Before pruning ===")
    pprint.pprint(model_sparsity_report(model))

    # apply structured fake pruning 30% of output channels/neurons
    structured_prune_model(model, amount=0.3, include_bias=False, verbose=True)

    print("\n=== After structured (masked) pruning (masks still present) ===")
    pprint.pprint(model_sparsity_report(model))

    # If you want to make the masks permanent (removes pruning bookkeeping)
    make_pruning_permanent(model)

    print("\n=== After removing pruning reparam (weights remain zeroed) ===")
    pprint.pprint(model_sparsity_report(model))

    # Example forward pass to verify model still runs
    x = torch.randn(2, 3, 32, 32).to(device)  # batch=2 image 32x32
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\nForward output shape: {out.shape}")