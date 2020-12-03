import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset
import numpy as np

use_gpu = torch.cuda.is_available()
file_root = '/mnt/d2/公共训练数据/JPEGImages/'

learning_rate = 0.001
num_epochs = 50
train_batch_size = 24
val_batch_size = 48

net = resnet50()

resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc'):
        print('yes')
        dd[k] = new_state_dict[k]

criterion = yoloLoss(7, 2, 5, 0.5)

print('Nick')



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

train_dataset = yoloDataset(root= file_root, list_file=['voc2012.txt', 'voc2007.txt'], train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
test_dataset = yoloDataset(root=file_root, list_file='voc2007test.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)

best_test_loss = np.inf

for epoch in range(num_epochs):
    torch.cuda.empty_cache()

    # training
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    elif epoch == 40:
        learning_rate = 0.00001

    total_loss = 0
    for i,(images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('train : Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (
            epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))

    # validation
    torch.cuda.empty_cache()
    validation_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images, target = images.cuda(), target.cuda()
            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.item()
            if (i + 1) % 5 == 0:
                # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0], total_loss / (i + 1)))
                print('test : Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (
                epoch + 1, num_epochs, i + 1, len(test_loader), loss.item(), validation_loss / (i + 1)))

        validation_loss /= len(test_loader)


    # save the best model
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'best.pth')

    torch.save(net.state_dict(), 'yolo.pth')
    torch.cuda.empty_cache()