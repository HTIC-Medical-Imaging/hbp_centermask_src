
import glob
import os
import numpy as np
from skimage.io import imread
# datadir='/Users/keerthi/Documents/work/htic/hbp/data/public/dsb2018/stage1_train'
from skimage.filters import unsharp_mask

class DSBdata:
    def __init__(self,img_dir='/data/public/dsb2018/stage1_train'):
        
        self.datadir = img_dir

        whitelist=[2,3,34,40,55,58,76,83,119,139,146,177,214,230,259,268,308,319,327,339,383,388,389,
                   416,430,449,463,473,496,515,592,611,613,620,625,637]

        fullimglist = os.listdir(self.datadir)
        
        self.imglist = [fullimglist[ii] for ii in whitelist]
        
        self.imgname = lambda n: glob.glob(self.datadir+'/'+self.imglist[n]+'/images/*.png')[0]

        self.msklist = lambda n: glob.glob(self.datadir+'/'+self.imglist[n]+'/masks/*.png')
        
        
    
    def __len__(self):
        return len(self.imglist)
        
    def load_image(self,imgno,with_mask=True, sharpen=True):
        
        imgpath = self.imgname(imgno)
        maskpaths = []
        if with_mask:
            maskpaths = self.msklist(imgno)

        img = imread(imgpath)[...,:3]
        if sharpen:
            img = unsharp_mask(img,preserve_range=True, channel_axis=2).astype(np.uint8)
            
        shp = img.shape
        msk_lbl = np.zeros((shp[0],shp[1]),np.uint16)


        for inst in range(len(maskpaths)):
            msk = maskpaths[inst]
            msk_i = imread(msk)
            msk_lbl[msk_i>0]=inst+1

        return img, msk_lbl

