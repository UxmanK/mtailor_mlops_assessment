import torch
from pytorch_model import Classifier
from torch.onnx import TrainingMode
import os
from torch import nn

class PreprocessModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x is expected to be in range [0, 1]
        x = (x - self.mean) / self.std
        return self.base_model(x)

def convert_to_onnx(output_path="/app/model.onnx"):
    # Initialize model
    base_model = Classifier()
    
    # Download weights if not present
    weights_path = "/app/pytorch_model_weights.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Please download weights and save as {weights_path}\n"
            "Original URL: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0"
        )
    
    # Load weights and set to eval mode
    base_model.load_state_dict(torch.load(weights_path, weights_only=True))
    base_model.eval()
    
    # Wrap with preprocessing
    model = PreprocessModel(base_model)
    model.eval()
    
    # Create example input (now expects input in [0,1] range)
    dummy_input = torch.randn(1, 3, 224, 224).clamp(0, 1)
    
    # Export to ONNX with preprocessing included
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        training=TrainingMode.EVAL
    )
    print(f"Model successfully converted to ONNX and saved at {output_path}")

if __name__ == "__main__":
    convert_to_onnx()