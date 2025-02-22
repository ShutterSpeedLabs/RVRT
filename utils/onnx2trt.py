import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def convert_onnx_to_trt(onnx_model_path, trt_model_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create a TensorRT builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    # Build the TensorRT engine
    builder.max_workspace_size = 1 << 30  # 1GB
    builder.max_batch_size = 1
    engine = builder.build_cuda_engine(network)
    
    # Serialize the engine to a file
    with open(trt_model_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"Model has been converted to {trt_model_path}")

if __name__ == "__main__":
    onnx_model_path = os.path.join(os.path.dirname(__file__), "../model_zoo_onnx/003_RVRT_videosr_bd_Vimeo_14frames.onnx")
    trt_model_path = os.path.join(os.path.dirname(__file__), "../model_zoo_trt/003_RVRT_videosr_bd_Vimeo_14frames.trt")
    convert_onnx_to_trt(onnx_model_path, trt_model_path)
    print(f"Model has been converted to {trt_model_path}")