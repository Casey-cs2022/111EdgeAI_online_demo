import numpy as np
import cv2
import os
from typing import Tuple
import io
# import tvm
# import tvm.relay
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
# import tvm.contrib.graph_runtime as graph_runtime
from mobilenet_v2_tsm_nl import MobileNetV2
from onnxsim import simplify
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print("Device :", device)

import argparse

parser = argparse.ArgumentParser(description="online demo setting")
parser.add_argument('--model', type = str, default = "")
parser.add_argument('--dataset', type = str, default = 'dmd')
parser.add_argument('--video', type = str, default = "")
parser.add_argument('--setting', type = str, default = 'video')

args = parser.parse_args()

model = args.model
dataset = args.dataset
video = args.video
setting = args.setting


SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

def get_net(use_gpu=True):

    if dataset == "dmd":
        torch_module = MobileNetV2(n_class=13)
    else:
        torch_module = MobileNetV2(n_class=2)

    torch_module.load_state_dict(rename_state_dict(os.path.join(model)))

    torch_module.eval()

    return torch_module.to(device)

def rename_state_dict(pth_path):
    '''
    rename the keys when load the model. 
    original use module.base.model.keynames to keynames
    '''
    pth = torch.load(pth_path)
    state_dict = pth['state_dict']
    new_state_dict = dict()
    for k, v in state_dict.items():
        if k.startswith('module.base_model.'):
            new_state_dict[k.replace('module.base_model.', '').replace('.net', '')] = v
        elif k.startswith('module.new_fc'):
            new_state_dict[k.replace('module.new_fc', 'classifier').replace('.net', '')] = v

    # for k, v in new_state_dict.items():
    #     print(k)

    return new_state_dict

def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform



if dataset == "dmd":
    catigories = [
        "Change Gear",
        "Drinking",
        "Hair and Makeup",
        "Phone call (Left)",
        "Phone call (Right)",
        "Radio",
        "Reach Backseat",
        "Reach Side",
        "Safe Driving",
        "Stand Still-Waiting",
        "Talking to passenger",
        "Texting (Left)",
        "Texting (Right)"  
    ]
else:
    catigories = [
        "ADL",
        "Fall"
    ]

n_still_frame = 0

def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 10  # max history buffer

    
    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Classification'
def main():

    if setting == "camera":
        print("Open camera...")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video)
    
    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("adl_1.avi", fourcc, 30.0, (640, 480))

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)


    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()
    # print("Build Executor...")
    net = get_net()
    shift_buffer = [
		torch.zeros([1, 3, 56, 56]).to(device),
		torch.zeros([1, 4, 28, 28]).to(device),
		torch.zeros([1, 4, 28, 28]).to(device),
		torch.zeros([1, 8, 14, 14]).to(device),
		torch.zeros([1, 8, 14, 14]).to(device),
		torch.zeros([1, 8, 14, 14]).to(device),
		torch.zeros([1, 12, 14, 14]).to(device),
		torch.zeros([1, 12, 14, 14]).to(device),
		torch.zeros([1, 20, 7, 7]).to(device),
		torch.zeros([1, 20, 7, 7]).to(device)
	]
    idx = 0
    history = [0, 0]
    history_logit = []

    i_frame = -1
    fsum = 0

    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        
        if i_frame % 1 == 0:  # skip every other frame to obtain a suitable frame rate

            start = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))  


            with torch.no_grad():
                feat, *shift_buffer = net(input_var.to(device), *shift_buffer)
                feat = feat.detach()


            if SOFTMAX_THRES > 0:
                feat = feat.cpu()
                feat_np = feat.numpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.numpy(), axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat.cpu().numpy(), axis=1)[0]


            if HISTORY_LOGIT:
                history_logit.append(feat.cpu().numpy())
                history_logit = history_logit[-20:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            print(f"{index} {catigories[idx]}")


            ftime = round(1 / (time.time() - start), 0)
            fsum += ftime

            FPS = round(fsum / (i_frame + 1), 0)


        img = cv2.resize(img, (640, 480))

        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)


        cv2.putText(label, '{:.1f} FPS'.format(FPS),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = cv2.vconcat([img, label])
        img = cv2.resize(img, (640, 480))

        cv2.imshow(WINDOW_NAME, img)

        out.write(img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    out.release()
    cap.release()
    cv2.destroyAllWindows()


main()
