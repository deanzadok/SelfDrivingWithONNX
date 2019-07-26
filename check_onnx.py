import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu\\model.onnx', type=str)
args = parser.parse_args()

# Load the ONNX model
model = onnx.load(args.path)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))