import sys
import os
import torch
import torch.onnx

# Add the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network_rvrt import RVRT as net

def convert_pth_to_onnx(model_path, onnx_path):
    # Load the model
    model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12, attention_heads=12,
                attention_window=[3, 3], nonblind_denoising=True, cpu_cache_length=100)
    
    # Load the pretrained weights
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 2, 64, 64)

    # Export the model
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "../model_zoo/rvrt/003_RVRT_videosr_bd_Vimeo_14frames.pth")
    onnx_path = os.path.join(os.path.dirname(__file__), "../model_zoo_onnx/003_RVRT_videosr_bd_Vimeo_14frames.onnx")
    convert_pth_to_onnx(model_path, onnx_path)
    print(f"Model has been converted to {onnx_path}")