from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


def write_label(x, img):
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    return label


if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'video.avi'

    cap = cv2.VideoCapture(0) #카메라1 (사람 detect하는 부분)
    cap2 = cv2.VideoCapture(1) #카메라2 (색깔 detect하는 부분)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():

        # time.sleep(0.5)
        # 이걸로 몇초마다 받을 지 설정 가능

        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            img2, orig_im2, dim2 = prep_image(frame2, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            #           im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            camera_detect_list = list(map(lambda x: write_label(x, orig_im), output))

            if 'person' in camera_detect_list:  # 이 부분에서 person이 detect되면 print 해준다.
                print('Person detected')

            cv2.imshow("frame", orig_im)

            #여기 부터 색깔 추출★★★★★★★★★★★★★★★★★★★★
            #★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            frame2 = cv2.resize(frame2, (100, 100))

            hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([110, 100, 100])
            upper_blue = np.array([130, 255, 255])

            lower_green = np.array([50, 100, 100])
            upper_green = np.array([70, 255, 255])

            lower_red = np.array([-10, 100, 100])
            upper_red = np.array([10, 255, 255])

            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_red = cv2.inRange(hsv, lower_red, upper_red)

            res1 = cv2.bitwise_and(frame2, frame2, mask=mask_blue)
            res2 = cv2.bitwise_and(frame2, frame2, mask=mask_green)
            res3 = cv2.bitwise_and(frame2, frame2, mask=mask_red)

            # blue_pixel_num = 0
            green_pixel_num = 0
            red_pixel_num = 0

            thresh_hold = 100

            res3_array = np.ravel(res3, order='C')
            for i in res3_array:
                if i>=1:
                    red_pixel_num=red_pixel_num+1
                if red_pixel_num >=thresh_hold:
                    break

            print("빨간 픽셀의 갯수 => ")
            print(red_pixel_num)

            if red_pixel_num >= thresh_hold:
                print("빨간불이 켜졌습니다.")

            res2_array = np.ravel(res2, order='C')
            for i in res2_array:
                if i >= 1:
                    green_pixel_num = green_pixel_num + 1
                if green_pixel_num >= thresh_hold:
                    break

            print("초록 픽셀의 갯수 => ")
            print(green_pixel_num)

            if green_pixel_num >= thresh_hold:
                print("초록불이 켜졌습니다.")

            cv2.imshow('original', frame2)
            # cv2.imshow('Blue', res1)
            cv2.imshow('Green', res2)
            cv2.imshow('Red', res3)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        else:
            break