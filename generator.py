import numpy as np
from data_gen import *
import fast_glcm
import nibabel as nib
import numpy as np #导入numpy模块
from PIL import Image #入PIL模块用于读取图片，也可使用opencv
import os
from CLAHEW import *
def AutoAdjustment(in_image):
    in_image = in_image*255
    a = 1.5
    out_image = float(a) * in_image
    # 进行数据截断, 大于255的值要截断为255
    out_image[out_image > 255] = 255
    # 数据类型转化
    out_image = np.round(out_image)
    out_image = out_image.astype(np.uint8)
    return out_image

def bn(image):
	[x,y,z] = np.shape(image)
	image = np.reshape(image,(x*y*z,1))
	max = np.max(image)
	min = np.min(image)
	image = (image-min)/(max-min)
	image = np.reshape(image,(x,y,z))
	return image
def clahm(image):  #d对图像进行均衡化
    image = image*255
    ones = np.ones((192, 192, 160))
    ones[image == 0] = 0
    image = np.asarray(image, dtype=np.uint8)
    img = np.ones((192,192,160))
    for i in range(160):
        image1 = image[:,:,i]
        image1 = clahe(image1)   #直方图均衡化
        img[:,:,i] = image1
    img = np.array(img,dtype=np.float32)
    img = img*ones
    img = bn(img)
    return img,ones
def generator_patch(x,y):
    x = np.reshape(x,(1,192,192,160))
    y = np.reshape(y,(1,192,192,160,5))
    x1,x2,x3,x4 = GLCM(x,160)
    train_patch = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64,64,64],
                                       stride_patch=[32,32,32], out=1, num_images=100)
    train_patch1 = vols_generator_patch(vol_name=x1, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch2 = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch3 = vols_generator_patch(vol_name=x3, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch4 = vols_generator_patch(vol_name=x4, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    #train_patch5 = vols_generator_patch(vol_name=x5, num_data=1, patch_size=[80,80,64],
    #                                     stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch6 = vols_generator_patch(vol_name=x6, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch7 = vols_generator_patch(vol_name=x7, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch8 = vols_generator_patch(vol_name=x8, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    mask_patch = vols_mask_generator_patch(vol_name=y, num_data=1, patch_size=[64,64,64,5],
                                           stride_patch=[32,32,32,5], out=1, num_images=100)
    return train_patch,train_patch1,train_patch2,train_patch3,train_patch4,mask_patch
def generator_patch_without_tumors(x,y):
    x = np.reshape(x,(1,192,192,160))
    y = np.reshape(y,(1,192,192,160,4))
    x1,x2,x3,x4 = GLCM(x,160)
    train_patch = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64,64,64],
                                       stride_patch=[32,32,32], out=1, num_images=100)
    train_patch1 = vols_generator_patch(vol_name=x1, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch2 = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch3 = vols_generator_patch(vol_name=x3, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    train_patch4 = vols_generator_patch(vol_name=x4, num_data=1, patch_size=[64,64,64],
                                        stride_patch=[32,32,32], out=1, num_images=100)
    #train_patch5 = vols_generator_patch(vol_name=x5, num_data=1, patch_size=[80,80,64],
    #                                     stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch6 = vols_generator_patch(vol_name=x6, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch7 = vols_generator_patch(vol_name=x7, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    #train_patch8 = vols_generator_patch(vol_name=x8, num_data=1, patch_size=[80,80,64],
    #                                      stride_patch=[40,40,32], out=1, num_images=75)
    mask_patch = vols_mask_generator_patch_without_tumors(vol_name=y, num_data=1, patch_size=[64,64,64,4],
                                           stride_patch=[32,32,32,4], out=1, num_images=100)
    return train_patch,train_patch1,train_patch2,train_patch3,train_patch4,mask_patch
def random(x_patch,x_patch1,x_patch2,x_patch3,x_patch4,x_mask):
    permutation = np.random.permutation(x_mask.shape[0])
    x_patch = x_patch[permutation, :, :] #训练数据
    x_patch1 = x_patch1[permutation, :, :] #训练数据
    x_patch2 = x_patch2[permutation, :, :] #训练数据
    x_patch3 = x_patch3[permutation, :, :] #训练数据
    x_patch4 = x_patch4[permutation, :, :] #训练数据
    x_mask = x_mask[permutation] #训练标签
    return x_patch,x_patch1,x_patch2,x_patch3,x_patch4,x_mask

def GLCM(image,num):
    image = np.reshape(image, (192,192,160))
    x1 = np.ones((1,192,192,160))
    x2 = np.ones((1,192,192,160))
    x3 =np.ones((1,192,192,160))
    x4 = np.ones((1,192,192,160))
    x5 =np.ones((1,192,192,160))
    x6 =np.ones((1,192,192,160))
    x7 =np.ones((1,192,192,160))
    x8 = np.ones((1,192,192,160))
    for i in range(num):
        img = image[:,:,i]
        img = np.reshape(img,(192,192))
        img = img*255
        mean = fast_glcm.fast_glcm_mean(img)
        std = fast_glcm.fast_glcm_std(img)
        cont = fast_glcm.fast_glcm_contrast(img)#对比度
        diss = fast_glcm.fast_glcm_dissimilarity(img)#相异度
        #homo = fast_glcm.fast_glcm_homogeneity(img) #同质性
        #asm, ene = fast_glcm.fast_glcm_ASM(img)
        #ma = fast_glcm.fast_glcm_max(img) #最大值
        #ent = fast_glcm.fast_glcm_entropy(img) #熵
        mean = mean/255
        std = std/255
        cont = cont/255
        diss =diss/255
       # homo = homo/255
        #asm=asm/255
        #ma = ma/255
        #ent =ent/255
        x1[:,:,:,i]=mean
        x2[:,:,:,i]=std
        x3[:,:,:,i]=cont
        x4[:,:,:,i]=diss
        #x5[...,i]=homo
        #x6[...,i]=asm
        #x7[...,i]=ma
        #x8[...,i]=ent
    return x1,x2,x3,x4
def generator_data(image_file,st,label_file,sl):
    vol_dir = image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_data()
    affine0 = image1.affine.copy()
    image = np.asarray(image, dtype=np.float32)
    image = np.reshape(image,(192,192,160))

    image,ones = clahm(image)
    label_dir = label_file+sl
    label = nib.load(label_dir).get_data()
    label = np.asarray(label, dtype=np.float32)
    affine0 = np.asarray(affine0, dtype=np.float32)
    return image,label,ones,affine0

def sig2(label):
    label = np.reshape(label,(192,192,160))
    BK = np.zeros((192,192,160))
    y_tumors = np.zeros((192,192,160))
    CSF = np.zeros((192,192,160))
    GM = np.zeros((192,192,160))
    WM = np.zeros((192,192,160))
    BK[label ==0] =1
    y_tumors[label ==1] =1
    CSF[label==2] =1
    GM[label ==3] =1
    WM[label ==4] =1
    BK = BK[...,np.newaxis]
    y_tumors = y_tumors[...,np.newaxis]
    CSF = CSF[...,np.newaxis]
    GM = GM[...,np.newaxis]
    WM = WM[...,np.newaxis]
    labels = np.concatenate([BK,y_tumors,CSF,GM,WM],axis= -1)
    labels_nomal = np.concatenate([BK,CSF,GM,WM],axis= -1)
    return labels,labels_nomal
def sig3(label):
    label = np.reshape(label,(192,192,160))
    BK = np.zeros((192,192,160))
    y_tumors = np.zeros((192,192,160))
    CSF = np.zeros((192,192,160))
    GM = np.zeros((192,192,160))
    WM = np.zeros((192,192,160))
    BK[label ==0] =1
    y_tumors[label ==1] =1
    y_tumors[label ==2] =1
    y_tumors[label ==4] =1
    CSF[label==5] =1
    GM[label ==6] =1
    WM[label ==7] =1
    BK = BK[...,np.newaxis]
    y_tumors = y_tumors[...,np.newaxis]
    CSF = CSF[...,np.newaxis]
    GM = GM[...,np.newaxis]
    WM = WM[...,np.newaxis]
    labels = np.concatenate([BK,y_tumors,CSF,GM,WM],axis= -1)
    return labels
from skimage import transform

def BatchGenerator_BraTS2020(train_file,train_mask_file,txt_file,txt_mask_file,batch_patch_size=1):
    while True:
        image_file = open(txt_file)  # 训练数据的名字放到txt文件里
        image_strings = image_file.readlines()
        mask_file = open(txt_mask_file)
        mask_strings = mask_file.readlines()
        for i in range(len(image_strings)):
            st = image_strings[i].strip()
            sl = mask_strings[i].strip()
            image,mask,ones,affine = generator_data(train_file,st,train_mask_file,sl)
            mask,mask_without_tumor = sig2(mask)
            x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch = generator_patch(image,mask)
            x_patch, x_patch1, x_patch2, x_patch3, x_patch4, y_patch = random(x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch)
            for start in range(0,len(x_patch),batch_patch_size):
                x_batch_patch = []
                x_batch_patch1 = []
                x_batch_patch2 = []
                x_batch_patch3 = []
                x_batch_patch4 = []
                y_batch_patch = []
                end = min(start + batch_patch_size, len(x_patch))
                for id in range(start, end):
                    img_patch = x_patch[id]
                    img_patch1 = x_patch1[id]
                    img_patch2 = x_patch2[id]
                    img_patch3 = x_patch3[id]
                    img_patch4 = x_patch4[id]
                    mask_patch = y_patch[id]
                    x_batch_patch.append(img_patch)
                    x_batch_patch1.append(img_patch1)
                    x_batch_patch2.append(img_patch2)
                    x_batch_patch3.append(img_patch3)
                    x_batch_patch4.append(img_patch4)
                    y_batch_patch.append(mask_patch)
                x_batch_patch = np.array(x_batch_patch)
                x_batch_patch1 = np.array(x_batch_patch1)
                x_batch_patch2 = np.array(x_batch_patch2)
                x_batch_patch3 = np.array(x_batch_patch3)
                x_batch_patch4 = np.array(x_batch_patch4)
                y_batch_patch = np.array(y_batch_patch)
                x_batch_patch = x_batch_patch[..., np.newaxis]
                x_batch_patch1 = x_batch_patch1[..., np.newaxis]
                x_batch_patch2 = x_batch_patch2[..., np.newaxis]
                x_batch_patch3 = x_batch_patch3[..., np.newaxis]
                x_batch_patch4 = x_batch_patch4[..., np.newaxis]
                y_tumors = y_batch_patch[..., 1:2]
                y_CSF = y_batch_patch[..., 2:3]
                y_GM = y_batch_patch[..., 3:4]
                y_WM = y_batch_patch[..., 4:5]
                yield ([x_batch_patch, x_batch_patch1, x_batch_patch2, x_batch_patch3, x_batch_patch4],
                       {'output8': y_batch_patch, 'output16': y_batch_patch, 'output32': y_batch_patch,
                        'output64': y_batch_patch, 'output1': y_batch_patch, 'output2': y_batch_patch,
                        'y_tumors': y_tumors, 'CSF': y_CSF, 'GM': y_GM, 'WM': y_WM})


def BatchGenerator_BraTS2020_other(train_file,train_mask_file,txt_file,txt_mask_file,batch_patch_size=1):
    while True:
        image_file = open(txt_file)  # 训练数据的名字放到txt文件里
        image_strings = image_file.readlines()
        mask_file = open(txt_mask_file)
        mask_strings = mask_file.readlines()
        for i in range(len(image_strings)):
            st = image_strings[i].strip()
            sl = mask_strings[i].strip()
            image,mask,ones,affine = generator_data(train_file,st,train_mask_file,sl)
            mask,mask_without_tumor = sig2(mask)
            x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch = generator_patch(image,mask)
            x_patch, x_patch1, x_patch2, x_patch3, x_patch4, y_patch = random(x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch)
            for start in range(0,len(x_patch),batch_patch_size):
                x_batch_patch = []
                x_batch_patch1 = []
                x_batch_patch2 = []
                x_batch_patch3 = []
                x_batch_patch4 = []
                y_batch_patch = []
                end = min(start + batch_patch_size, len(x_patch))
                for id in range(start, end):
                    img_patch = x_patch[id]
                    img_patch1 = x_patch1[id]
                    img_patch2 = x_patch2[id]
                    img_patch3 = x_patch3[id]
                    img_patch4 = x_patch4[id]
                    mask_patch = y_patch[id]
                    x_batch_patch.append(img_patch)
                    x_batch_patch1.append(img_patch1)
                    x_batch_patch2.append(img_patch2)
                    x_batch_patch3.append(img_patch3)
                    x_batch_patch4.append(img_patch4)
                    y_batch_patch.append(mask_patch)
                x_batch_patch = np.array(x_batch_patch)
                x_batch_patch1 = np.array(x_batch_patch1)
                x_batch_patch2 = np.array(x_batch_patch2)
                x_batch_patch3 = np.array(x_batch_patch3)
                x_batch_patch4 = np.array(x_batch_patch4)
                y_batch_patch = np.array(y_batch_patch)
                x_batch_patch = x_batch_patch[..., np.newaxis]
                x_batch_patch1 = x_batch_patch1[..., np.newaxis]
                x_batch_patch2 = x_batch_patch2[..., np.newaxis]
                x_batch_patch3 = x_batch_patch3[..., np.newaxis]
                x_batch_patch4 = x_batch_patch4[..., np.newaxis]
                y_tumors = y_batch_patch[..., 1:2]
                y_CSF = y_batch_patch[..., 2:3]
                y_GM = y_batch_patch[..., 3:4]
                y_WM = y_batch_patch[..., 4:5]
                yield ([x_batch_patch],
                       { 'output': y_batch_patch,'y_tumors': y_tumors, 'CSF': y_CSF, 'GM': y_GM, 'WM': y_WM})

def BatchGenerator_tumors_2D(train_file, train_mask_file, txt_file, txt_mask_file, batch_patch_size=1):
    while True:
        image_file = open(txt_file)  # 训练数据的名字放到txt文件里
        image_strings = image_file.readlines()
        mask_file = open(txt_mask_file)
        mask_strings = mask_file.readlines()
        for i in range(len(image_strings)):
            st = image_strings[i].strip()
            sl = mask_strings[i].strip()
            image,mask,ones,affine = generator_data(train_file,st,train_mask_file,sl)
            mask,mask_without_tumor = sig2(mask)
            # x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch = generator_patch(image,mask)
            x_patch = image
            y_patch = mask
            for start in range(0, len(x_patch), batch_patch_size):
                x_batch_patch = []
                y_batch_patch = []
                end = min(start + batch_patch_size, len(x_patch))
                for id in range(start, end):
                    img_patch = x_patch[id]
                    mask_patch = y_patch[id]
                    x_batch_patch.append(img_patch)
                    y_batch_patch.append(mask_patch)
                x_batch_patch = np.array(x_batch_patch)
                y_batch_patch = np.array(y_batch_patch)
                x_batch_patch = x_batch_patch[..., np.newaxis]
                y_tumors = y_batch_patch[..., 1:2]
                y_CSF = y_batch_patch[..., 2:3]
                y_GM = y_batch_patch[..., 3:4]
                y_WM = y_batch_patch[..., 4:5]
                yield ([x_batch_patch],
                       {'output': y_batch_patch, 'y_tumors': y_tumors, 'CSF': y_CSF, 'GM': y_GM, 'WM': y_WM})