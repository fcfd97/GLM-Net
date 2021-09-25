from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from data import *
from generator import BatchGenerator_BraTS2020
from loss import my_loss1,dice_coef,IoU,csf_loss,my_loss_mutil_out
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from keras.optimizers import Adam

#from my_attention_model import ResNetR3_attention_mutil_scale

from my_model import ResNetR3_attention_mutil_scale
from keras.callbacks import ModelCheckpoint,EarlyStopping
learn_rate = 1e-4
train_file = 'train_data/train/'
train_mask_file = 'train_data/train_mask/'
train_txt = 'train_data/txt_file/train.txt'
train_mask_txt = 'train_data/txt_file/train_mask.txt'
valid_file = 'train_data/valid/'
valid_mask_file = 'train_data/valid_mask/'
valid_txt = 'train_data/txt_file/valid.txt'
valid_mask_txt = 'train_data/txt_file/valid_mask.txt'
myGene = BatchGenerator_BraTS2020(train_file,train_mask_file,train_txt,train_mask_txt, 1)
myvaild_Gene = BatchGenerator_BraTS2020(valid_file,valid_mask_file,valid_txt,valid_mask_txt, 1)
#model = ResNetR3_attention((80,80,64,1))

model = ResNetR3_attention_mutil_scale((64,64,64,1), (64,64,64,1),(64,64,64,1),(64,64,64,1),(64,64,64,1))
model.summary()
model.compile(optimizer=Adam(lr=learn_rate), loss={'output8':my_loss1,'output16':my_loss1,'output32':my_loss1,'output64':my_loss1,'output1':my_loss1,'output2':my_loss1,'y_tumors':csf_loss,'CSF':csf_loss,'GM':csf_loss,'WM':csf_loss},metrics=[dice_coef])
model_checkpoint = ModelCheckpoint('my_1.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint1 = ModelCheckpoint('weight/weight_tu/weights.{epoch:02d}-{loss:.4f}.hdf5',monitor='loss',verbose = 1,save_best_only=True,save_weights_only=True,mode='auto',period=1)
early_stop = EarlyStopping(monitor='loss',patience=10)
model.fit_generator(myGene,steps_per_epoch=28900,epochs=20,validation_data=myvaild_Gene,validation_steps = 8000,callbacks=[model_checkpoint,model_checkpoint1,early_stop])
model.save('result/my_model_1.h5')