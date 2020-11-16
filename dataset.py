'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import torch.utils.data as data
import os
import torchvision.transforms as transforms

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

        a = 10



def main():
    file_root = '/mnt/d2/公共训练数据/JPEGImages/'
    train_dataset = yoloDataset(root=file_root, list_file=['voc2012.txt', 'voc2007.txt'], train=True, transform=[transforms.ToTensor()])
if __name__ == '__main__':
    main()