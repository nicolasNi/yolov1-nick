import os
import numpy as np
import cv2
from torchvision import models
import torch
from torchsummary import summary
from resnet_yolo import resnet50, resnet18


# # 这个是你图片的根目录，注意不要带中文路径，楼主就因为这个调试了很久。
# image_path = '/train/image'
# image_path = '/mnt/d2/公共训练数据/JPEGImages/'
#
#
# file_names = os.listdir(image_path)
#
# count = 0
# mean = np.zeros(3, np.int64)
#
#
# for i in file_names[1:]:
#     # print(i)
#     print(count)
#
#     img = cv2.imread(image_path + '/' + i)
#     #print(img)
#     count += 1
#     mean += np.sum(img, axis=(0, 1)).astype(int)
# h, w = img.shape[0:-1]
# print(h, w, count)
# means = mean / (1.0 * h * w * count)
# print('b, g, r = ', means)


#
# new_state_dict = resnet.state_dict()
# for k in new_state_dict.keys():
#     print(k)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

#
# net = resnet50().to(device)
# resnet = models.resnet50(pretrained=True).to(device)
# summary(resnet,(3, 224, 224))


net = models.resnet18().to(device)
summary(net,(3,224,224))
# print(net)