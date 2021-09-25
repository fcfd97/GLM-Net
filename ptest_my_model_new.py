from write_excel import *
from data import *
from generator import *
from loss import *
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from keras.optimizers import Adam
import cv2
from metrics import mutil_accuracy,mutil_asd,mutil_hd
from my_model import ResNetR3_attention_mutil_scale
learn_rate = 1e-4
def generator_test_patch(x):
    x = x[np.newaxis,...]
    x1,x2,x3,x4 = GLCM(x,160)
    train_patch , train_loc = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=2, num_images=100)
    train_patch1,a = vols_generator_patch(vol_name=x1, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=2, num_images=100)
    train_patch2,b = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=2, num_images=100)
    train_patch3,c = vols_generator_patch(vol_name=x3, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=2, num_images=100)
    train_patch4,d = vols_generator_patch(vol_name=x4, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=2, num_images=100)
    return train_patch,train_loc,train_patch1,train_patch2,train_patch3,train_patch4
#
model = ResNetR3_attention_mutil_scale((64,64,64,1), (64,64,64,1),(64,64,64,1),(64,64,64,1),(64,64,64,1))
weight_file = open( 'weight/weight_tu/weight.txt')  # 训练数据的名字放到txt文件里
weight_strings = weight_file.readlines()
valid_file = open('train_data/txt_file/test.txt')
valid_strings = valid_file.readlines()
valid_mask_file = open('train_data/txt_file/test_mask.txt')
valid_mask_strings = valid_mask_file.readlines()
for k in range(0,20):
    st = weight_strings[k].strip()  # 文件名
    weight = 'weight/weight_tu/'+st
    model.load_weights(weight,by_name=True)
    for j in range(len(valid_strings)):
        mask = np.empty((192,192,160,5))
        mask2 = np.empty((192,192,160,5))
        st = valid_strings[j].strip()  # 文件名
        sl = valid_mask_strings[j].strip()
        y_ones = np.ones((192,192,160))
        x_test,y_test,ones, affine1 = generator_data('train_data/test/',st,'train_data/test_mask/',sl)
        #x_test = x_test*y_ones
        test_vols, test_vols_loc, test_vols1, test_vols2, test_vols3, test_vols4 = generator_test_patch(x_test)
        pred = []
        for i in range(len(test_vols[0])):
            pred_temp = model.predict([test_vols[0][i],test_vols1[0][i],test_vols2[0][i],test_vols3[0][i],test_vols4[0][i]])
        #pred_temp = model.predict(test_vols[j][i])
            pred_temp1 = pred_temp[5]
            mask[test_vols_loc[0][i][0].start:test_vols_loc[0][i][0].stop,
            test_vols_loc[0][i][1].start:test_vols_loc[0][i][1].stop,
		    test_vols_loc[0][i][2].start:test_vols_loc[0][i][2].stop,:] += pred_temp1[0,:,:,:,:]
            mask2[test_vols_loc[0][i][0].start:test_vols_loc[0][i][0].stop,
            test_vols_loc[0][i][1].start:test_vols_loc[0][i][1].stop,
		    test_vols_loc[0][i][2].start:test_vols_loc[0][i][2].stop,:] += np.ones(pred_temp1.shape[1:]).astype('float32')
        pred_mask = mask / mask2
        mask3 = np.argmax(pred_mask, axis=-1)
        x_test = np.reshape(x_test, (192, 192, 160))
        # nib.save(nib.Nifti1Image(x_test, affine1), 'predict/propose/test/' + str(j) + '_test.nii.gz')
        # nib.save(nib.Nifti1Image(y_test, affine1),'predict/propose/mask/'+str(j)+'_label.nii.gz')
        pred_image = best_map(y_test, mask3, [192, 192, 160])
        nib.save(nib.Nifti1Image(pred_image,affine1),'predict/pre/'+str(k)+'_'+str(j)+'_predict.nii.gz')
        pred.append(pred_mask)
        val = dice(y_test, pred_image)
        val_mean = np.mean(val)
        dice1 = [[val[0],val[1], val[2],val[3], np.mean([val[1], val[2],val[3]]),val_mean], ]
        print(dice1)
        path1='predict/propose/dice.xls'
        write_excel_xls_append(path1,dice1)

        acc, acc_bk = mutil_accuracy(y_test, pred_image)
        print(acc)
        write_excel_xls_append('predict/propose/acc.xls', acc)

        asd, asd_bk = mutil_asd(y_test, pred_image)
        print(asd)
        write_excel_xls_append('predict/propose/asd.xls', asd)

        hd, hd_bk = mutil_hd(y_test, pred_image)
        print(hd)
        write_excel_xls_append('predict/propose/hd.xls', hd)





