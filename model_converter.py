import argparse
import os.path
import sys
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from google.protobuf.json_format import MessageToDict
from onnx_tf.backend import prepare

from preprocessing.magface import iresnet


class NetworkBuilder_inf(nn.Module):
    def __init__(self, args):
        super(NetworkBuilder_inf, self).__init__()
        self.features = iresnet.iresnet100(pretrained=False,
                                           num_classes=args.embedding_size)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--img_size', required=True)
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_dict_inf(args, model):
    if os.path.isfile(args.resume):
        print('=> loading pth from {} ...'.format(args.resume))
        if args.cpu_mode:
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(args.resume)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args.resume))
    return model


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.' + '.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
                v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
                v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


if __name__ == '__main__':
    args = arg_parser()
    torch_model_path = args.model_path + '.pth'
    onnx_model_path = args.model_path + '.onnx'

    dummy = torch.randn(64, 3, args.img_size, args.img_size)

    Torch_args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    torch_args = Torch_args('iresnet100', torch_model_path, 512, True)

    torch_model = NetworkBuilder_inf(torch_args)
    torch_model = load_dict_inf(torch_args, torch_model)

    torch_model.eval()
    torch_out = torch_model(dummy)
    if not os.path.exists(onnx_model_path):
        torch.onnx.export(torch_model, dummy, onnx_model_path)
        print("Torch model converted to ONNX and saved to file!")
    else:
        print("Torch model was already converted to ONNX!")

    # ONNX
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked and it is fine!")

    print("ONNX graph configurations:")
    for _input in onnx_model.graph.input:
        print(MessageToDict(_input))
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported ONNX model has been tested with ONNXRuntime, and the result looks good!")

    # TensorFlow
    print("Converting the ONNX model to TF:")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(args.model_path + "_pb")
    print("TF model saved!")
