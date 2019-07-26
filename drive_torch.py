import sys
sys.path.append('.')
import airsim
import os
import time
import numpy as np
import argparse
import torch
from model import ILModel
from PIL import Image
import cv2
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu\\model_30.pt', type=str)
parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present front camera on screen', action='store_true')
parser.add_argument('--fps_mean', '-fps_mean', help='number of fps vals to compute the mean over', default=100, type=int)
args = parser.parse_args()

if __name__ == "__main__":

    # define gpu as device
    device = torch.device("cuda")

    # Load imitation model
    model = ILModel(num_actions=args.num_actions).to(device)
    model.load_state_dict(torch.load(args.path))

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # actions list
    actions = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # let the car start driving
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)

    print('Running model')

    # define camera image request
    Image_requests = [airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)]

    # fps calculation
    i = 0
    fps_q = np.zeros((args.fps_mean))

    while(True):

        start_time = time.time()

        # get images from AirSim
        responses = client.simGetImages(Image_requests)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
            image = Image.fromarray(img2d).convert('L')
            image_np = np.array(image.resize((84, 84)))
        else:
            image_np = np.zeros((84,84)).astype(float)

        # present state image if debug mode is on
        if args.debug:
            cv2.imshow('navigation map', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # normalize, convert input to [1,1,84,84] and convert to torch tensor
        input_img = np.expand_dims(np.expand_dims(image_np, axis=0), axis=0).astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_img).to(device)

        # predict steering action
        action_values = model(input_tensor)
        car_controls.steering = actions[np.argmax(action_values.detach().cpu().numpy())]

        # send action to AirSim
        client.setCarControls(car_controls)

        end_time = time.time()
        # store fps  
        fps_q[i%args.fps_mean] = 1/(end_time-start_time)
        i+=1

        print('steering = {}, fps = {}'.format(car_controls.steering,fps_q.mean()))

        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            sys.exit()