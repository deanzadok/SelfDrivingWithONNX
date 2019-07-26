import sys
sys.path.append('.')
import airsim
import os
import time
import h5py
import numpy as np
import argparse
import torch
import onnxruntime as rt
from model import ILModel
from PIL import Image
import cv2
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', '-data_file', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\gs_4images_nocov\\gs_4images_nocov.h5', type=str)
parser.add_argument('--onnx_path', '-onnx_path', help='onnx model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu\\model.onnx', type=str)
parser.add_argument('--torch_path', '-torch_path', help='torch model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu\\model_30.pt', type=str)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to evaluate on', default=5000, type=int)
parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
parser.add_argument('--fps_mean', '-fps_mean', help='number of fps vals to compute the mean over', default=100, type=int)
args = parser.parse_args()

def load_dataset(dset_file, num_imgs):

    # load h5 file
    dataset_dict = h5py.File(dset_file, 'r')

    # get dataset as numpy
    images_dataset = np.asarray(dataset_dict['images'], dtype=np.float32)
    labels_dataset = np.asarray(dataset_dict['labels'], dtype=np.int)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_dataset.shape[0])
    images_dataset = images_dataset[p]
    labels_dataset = labels_dataset[p]

    # NHWC => NCHW
    images_dataset = images_dataset.transpose(0,3,1,2)

    # trim data if asked to
    if images_dataset.shape[0] > num_imgs:
        images_dataset = images_dataset[:num_imgs,:,:,:]
        labels_dataset = labels_dataset[:num_imgs,:]

    return images_dataset, labels_dataset

if __name__ == "__main__":

    # define gpu as device
    device = torch.device("cuda")

    # Load torch model
    model = ILModel(num_actions=args.num_actions).to(device)
    model.load_state_dict(torch.load(args.torch_path))

    # Load onnx model using onnx runtime
    sess = rt.InferenceSession(args.onnx_path)

    # get the name of the first input of the model
    input_name = sess.get_inputs()[0].name
    print('Input Name: {}'.format(input_name))

    # load images and labels
    images, labels = load_dataset(args.data_file, args.num_imgs)

    # initiate numpy arrays to evaluate fps
    fps_torch, fps_onnx = np.zeros((images.shape[0])), np.zeros((images.shape[0]))

    # initiate numpy arrays to evaluate acuracy
    accuracy_torch = np.zeros((images.shape[0]), dtype=np.int32)
    accuracy_onnx = np.zeros((images.shape[0]), dtype=np.int32)
    agreement = np.zeros((images.shape[0]), dtype=np.int32)

    # iterate images and evaluate performance of both models
    for i in range(images.shape[0]):
        
        # filter input image
        input_img = np.expand_dims(images[i], axis=0)
        
        # inference using torch and measure time
        input_tensor = torch.tensor(input_img).to(device)
        start_time = time.time()
        torch_values = model(input_tensor)
        end_time = time.time()
        fps_torch[i] = (end_time-start_time)
        torch_values = torch_values.detach().cpu().numpy()

        # inference using onnx and measure time
        start_time = time.time()
        onnx_values = sess.run([], {input_name: input_img})
        end_time = time.time()
        fps_onnx[i] = (end_time-start_time)
        onnx_values = onnx_values[0]

        # evaluate both models
        accuracy_torch[i] = 1 if np.argmax(torch_values) == labels[i][0] else 0
        accuracy_onnx[i] = 1 if np.argmax(onnx_values) == labels[i][0] else 0
        agreement[i] = 1 if np.argmax(onnx_values) == np.argmax(torch_values) else 0

        if i % 100 == 0:
            print('{}/{} images inferred'.format(i,images.shape[0]))
    
    print("* ---------------------- *")
    print("FPS: torch - {:.2f} in average. onnx - {:.2f} in average.".format(1/fps_torch.mean(), 1/fps_onnx.mean()))
    print("Accuracy: torch - {:.2f}%. onnx - {:.2f}%.".format((accuracy_torch.sum()/args.num_imgs)*100.0, (accuracy_onnx.sum()/args.num_imgs)*100.0))
    print("Agreement between models: {:.2f}%.".format((agreement.sum()/args.num_imgs)*100.0))

