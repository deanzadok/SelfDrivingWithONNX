import os
import argparse
import torch
from model import ILModel

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu\\model_30.pt', type=str)
parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
args = parser.parse_args()

if __name__ == "__main__":

    # define gpu as device
    device = torch.device("cuda")

    # Load imitation model
    model = ILModel(num_actions=args.num_actions).to(device)
    model.load_state_dict(torch.load(args.path))

    # define input tensor for example
    dummy_input = torch.randn(1, 1, 84, 84, device='cuda')

    # save onnx model to dest dir
    dest_dir = os.path.dirname(args.path)
    torch.onnx.export(model, dummy_input, os.path.join(dest_dir,"model.onnx"))