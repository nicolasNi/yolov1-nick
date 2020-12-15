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
from tensorboardX import SummaryWriter
from tqdm import tqdm
from predict import *
from collections import defaultdict


VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            break
        # print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]  # 预测框
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]  # [[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                                bbgt[3] - bbgt[1] + 1.) - inters
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # 这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id, class_)]  # 删除没有box的键值
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))
    return np.mean(aps)


write = SummaryWriter(comment='nick-yolo1')
use_gpu = torch.cuda.is_available()
file_root = '/mnt/d2/公共训练数据/JPEGImages/'
file_root = '/mnt/wwn-0x5000039fe6f44156-part3/公共训练数据/JPEGImages/'

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
            write.add_scalar('trainingLoss', total_loss / (i + 1), epoch * len(train_loader) + i)

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
                write.add_scalar('validationLoss', validation_loss / (i + 1), epoch * len(test_loader) + i)
        validation_loss /= len(test_loader)


    # evaluation
    torch.cuda.empty_cache()
    target = defaultdict(list)
    preds = defaultdict(list)
    image_list = []  # image path list

    f = open('voc2007test.txt')
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()
    print('---prepare target---')
    for index, image_file in enumerate(file_list):
        image_id = image_file[0]

        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = VOC_CLASSES[c]
            target[(image_id, class_name)].append([x1, y1, x2, y2])
    net.cuda()
    count = 0
    for image_path in tqdm(image_list):
        file_root = '/mnt/wwn-0x5000039fe6f44156-part3/公共训练数据/JPEGImages/'
        result = predict_gpu(net, image_path,
                             root_path=file_root)  # result[[left_up,right_bottom,class_name,image_path],]
        for (x1, y1), (x2, y2), class_name, image_id, prob in result:  # image_id is actually image_path
            preds[class_name].append([image_id, prob, x1, y1, x2, y2])
    map = voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES)
    write.add_scalar('map', map)
    print("the map is " + map)

    # save the best model
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'best.pth')

    torch.save(net.state_dict(), 'yolo.pth')
    torch.cuda.empty_cache()