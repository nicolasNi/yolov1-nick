import os
import numpy as np
import cv2

# 这个是你图片的根目录，注意不要带中文路径，楼主就因为这个调试了很久。
image_path = '/train/image'
image_path = '/mnt/d2/公共训练数据/JPEGImages/'


file_names = os.listdir(image_path)

count = 0
mean = np.zeros(3, np.int64)


for i in file_names[1:]:
    # print(i)
    print(count)

    img = cv2.imread(image_path + '/' + i)
    #print(img)
    count += 1
    mean += np.sum(img, axis=(0, 1)).astype(int)
h, w = img.shape[0:-1]
print(h, w, count)
means = mean / (1.0 * h * w * count)
print('b, g, r = ', means)


