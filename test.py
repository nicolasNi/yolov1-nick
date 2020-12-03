import os
import numpy as np
import cv2
from torchvision import models
import torch
from torchsummary import summary
from resnet_yolo import resnet50, resnet18
from torch.autograd import Variable

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


net = resnet50().to(device)
resnet = models.resnet50(pretrained=True).to(device)
# summary(net,(3, 448, 448))
# print(net)

new_state_dict = resnet.state_dict()
dd = net.state_dict()
# for k in dd.keys():
#     print(k)


# net = models.resnet50().to(device)
# # summary(net,(3,224,224))
# summary(net,(3,448,448))
# # print(net)

#
# params = []
# params_dict = dict(resnet.named_parameters())
# for key, value in params_dict.items():
#     print(key)



w1 = torch.Tensor([2]) #认为w1 与 w2 是函数f1 与 f2的参数
w1 = Variable(w1,requires_grad=True)
w2 = torch.Tensor([2])
w2 = Variable(w2,requires_grad=True)
x2 = torch.rand(1)
x2 = Variable(x2,requires_grad=True)
y2 = x2**w1            # f1 运算
z2 = w2*y2+1           # f2 运算
z2.backward()

print(w1)
print(w2)


print(x2.grad)
print(y2.grad)
print(w1.grad)
print(w2.grad)
