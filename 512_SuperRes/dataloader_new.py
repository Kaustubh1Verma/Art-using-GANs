import scipy
from glob import glob
import numpy as np
import os

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        #path1 = glob('/users/home/dlagroup4/project/AOD/NYU/small_haze/'+'*.jpg')
        #path2 = '/users/home/dlagroup4/project/AOD/NYU/small_gt/'

        path1 = glob('/users/home/b16004/wd/Dataset/Data/NewData/512/'+'*.png')
        path2 = '/users/home/b16004/wd/Dataset/Data/GT/512/'

        #print(len(path1))
        #batch_images1 = path1
        batch_images1 = np.random.choice(path1, size=batch_size)
	
        imgs_A = []
        imgs_B = []

        imgs_A_or = []
        imgs_B_or = []

        for img_path in batch_images1:
            img_gt = self.imread(img_path)
            img_haze = self.imread(path2 + img_path[47:])

            imgs_A_or.append(img_gt)
            imgs_B_or.append(img_haze)

            h, w, _ = img_gt.shape
            img_A, img_B = img_haze,img_gt

            #img_A = scipy.misc.imresize(img_A, self.img_res)
            #img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B, np.array(imgs_B_or)/127.5 - 1, np.array(imgs_A_or)/127.5 - 1

    def load_batch(self, batch_size=1, is_testing=False):
      
        # path1 = glob('/users/home/dlagroup4/project/AOD/NYU/train/haze/'+'*.jpg')
        # path2 = '/users/home/dlagroup4/project/AOD/NYU/train/gt/'

        path1 = glob('/users/home/b16004/wd/Dataset/Data/NewData/512/'+'*.png')
        path2 = '/users/home/b16004/wd/Dataset/Data/GT/512/'
        
        #print(path1) 
        self.n_batches = int(len(path1) / batch_size)

        for i in range(self.n_batches-1):
            batch1 = path1[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch1:
                img_gt = self.imread(img)
                img_haze = self.imread(path2 + img[47:])

                h, w, _ = img_gt.shape
                img_A,img_B = img_haze,img_gt

                #img_A = scipy.misc.imresize(img_A, self.img_res)
                #img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
