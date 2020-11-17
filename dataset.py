'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import torch.utils.data as data
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
import cv2
import random
import numpy as np
import torch

class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(self, root, list_file, train, transform):
        print('data init')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.Tensor(label))
        self.num_sample = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        self.show_img(img, boxes, 'original')

        if self.train:
            # img, boxes = self.random_flip(img, boxes)
            # img, boxes = self.random_scale(img, boxes)
            # img = self.random_blur(img)
            # img = self.random_brightness(img)
            # img = self.random_hue(img)
            # img = self.random_saturation(img)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            self.show_img(img, boxes, 'random_shift')
            a = 10

    def __len__(self):
        return self.num_sample

    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(img).copy()
            h, w, _ = im_lr.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return img, boxes

    def random_scale(self, img_bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scalew = random.uniform(0.8, 1.2)
            scaleh = random.uniform(0.8, 1.2)
            height, width, c = img_bgr.shape
            # img_bgr = cv2.resize(img_bgr, (int(width * scale), height))
            # scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            img_bgr = cv2.resize(img_bgr, (int(width * scalew), int(height * scaleh)))
            scale_tensor = torch.FloatTensor([[scalew, scaleh, scalew, scaleh]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return img_bgr, boxes
        else:
            return img_bgr, boxes

    def random_blur(self, img):
        if random.random() < 0.5:
            img = cv2.blur(img, (5, 5))
        return img

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def random_brightness(self, img):
        if random.random() < 0.5:
            img_hsv = self.BGR2HSV(img)
            h, s, v = cv2.split(img_hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h, s, v))
            img = self.HSV2BGR(img_hsv)
        return img

    def random_hue(self, img):
        if random.random() < 0.5:
            img_hsv = self.BGR2HSV(img)
            h, s, v = cv2.split(img_hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(v, 0, 179).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h, s, v))
            img = self.HSV2BGR(img_hsv)
        return img

    def random_saturation(self, img):
        if random.random() < 0.5:
            img_hsv = self.BGR2HSV(img)
            h, s, v = cv2.split(img_hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(v, 0, 255).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h, s, v))
            img = self.HSV2BGR(img_hsv)
        return img

    def random_shift(self, bgr, boxes, labels):
        # if random.random() < 0.5:
        center = (boxes[:, 2:] + boxes[:,:2]) / 2
        height, width, c = bgr.shape
        after_shift_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shift_image[:, :, :] = (104, 117, 123)
        shift_x = width * 0.2
        shift_y = height * 0.2
        # shift_x = random.uniform(-width * 0.2, width * 0.2)
        # shift_y = random.uniform(-height * 0.2, height * 0.2)

        after_shift_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), : width - int(shift_x), :]

        shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
        center += shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
        boxes_in += box_shift
        labels_in = labels[mask.view(-1)]
        return after_shift_image, boxes_in, labels_in

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        # if random.random() < 0.5:
        if True:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def show_img(self, img, boxes, name):
        im = img.copy()
        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        for box in boxes:
            point1 = (box[0], box[1])
            point2 = (box[2], box[3])
            cv2.rectangle(im, point1, point2, point_color, thickness, lineType)
        cv2.imshow(name, im)
        cv2.waitKey()


def main():
    file_root = '/mnt/d2/公共训练数据/JPEGImages/'
    train_dataset = yoloDataset(root=file_root, list_file=['voc2012.txt', 'voc2007.txt'], train=True, transform=[transforms.ToTensor()])
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
if __name__ == '__main__':
    main()