import torch
from resnet_yolo import resnet50
from net import vgg16_bn
from torchvision import models
import yoloLoss

use_gpu = torch.cuda.is_available()
file_root = '/mnt/d2/公共训练数据/JPEGImages/'

learning_rate = 0.001
num_epochs = 50
train_batch_size = 24
val_batch_size = 48
use_resnet = True
if use_resnet:
    net = resnet50()
else:
    net = vgg16_bn()
print(net)


print("load pre-trained model:")
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc'):
        print('yes')
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)

print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7, 2, 5, 0.5)
if use_gpu:
    net.cuda()

net.train()

# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)



print(use_gpu)