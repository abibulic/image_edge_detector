import io
import models
import argparse
import numpy as np

import torch.onnx
from torch import nn

import onnx
import onnxruntime
from onnx import optimizer

import models
from vgg import VggHed, load_vgg16pretrain

def model_to_onnx(args):
    
    if 'vgg' in args.backbone_model:
        torch_model = VggHed(args)
        checkpoint = torch.load(args.onnx_model, map_location=torch.device('cpu'))

        temp_dict = checkpoint['state_dict'].copy()
        for i, s in enumerate(temp_dict):
            temp = s.replace('model.','')
            checkpoint['state_dict'][temp] = checkpoint['state_dict'].pop(s)
        
    else:
        torch_model = models.HED(args)
        checkpoint = torch.load(args.onnx_model, map_location=torch.device('cpu'))

    torch_model.load_state_dict(checkpoint['state_dict'])
    torch_model.eval()
     
    x = torch.rand(args.batch_size, args.n_channels, args.input_img_size_x, args.input_img_size_y)
    torch_out = torch_model(x)

    onnx_model_file = args.onnx_model.split('/')
    temp_file = str()
    for o in onnx_model_file[:-1]:
        temp_file += o
        temp_file += '/'
    onnx_model_file = temp_file + "edge_detector_model.onnx"

    print("Exporting model...")

    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_model_file,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                   'output' : {0 : 'batch_size'}})

    


    # ----------- TODO: Call to optimizer.optimize below finishes the script so this is commented out for now.
    # print("Loading exported model for optimization...")

    # onnx_model = onnx.load(onnx_model_file)
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]

    # print("Optimizing...")
    # optimized_model = optimizer.optimize(onnx_model, passes)

    # onnx.save(optimized_model, onnx_model_file)

    print("Checking model...")

    onnx_model = onnx.load(onnx_model_file)
    onnx.checker.check_model(onnx_model)

    print("Comparing out with torch output")

    ort_session = onnxruntime.InferenceSession(onnx_model_file)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    #np.testing.assert_allclose(to_numpy(torch_out[2]), ort_outs[2], rtol=1e-03, atol=1e-05)
    
    print("Exported model has been tested with ONNXRuntime and compared to pytorch, and the result looks good!")