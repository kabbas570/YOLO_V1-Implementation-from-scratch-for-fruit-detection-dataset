import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import optimizers

import numpy as np
import tensorflow.keras as keras
from  read_data_youl_format import data_read
from  read_data_youl_format import data_readV


from loss import yolo_loss
#from model1 import model_tiny_yolov1
from VGG_YOLO import VGG_16


images,target=data_read()
imagesV,targetV=data_readV()


model=VGG_16()
# summarize
print( model.layers)
for layer in model.layers[:19]:
    layer.trainable=False
model.summary()
model.summary()


epochs=5
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model.compile(optimizer=Adam, loss=yolo_loss)
model.fit(images,target,validation_data=(imagesV,targetV),batch_size=8, 
                    epochs=epochs)


#model.save_weights("yolo_fruits_vgg.h5")

model.load_weights("yolo_fruits_vgg.h5")
predictions=model.predict(imagesV)


from plot_ import _main
#plot predictions coordinates on images
for i in range(160,170):
    image_id=i
    ig=predictions[i,:,:,:].copy()
    ig=np.expand_dims(ig,axis=0)
    img=imagesV[image_id,:,:,:].copy()
    a=_main(ig,img,i)



plt.figure()
plt.imshow(images[2020,:,:,:])
from plot_tar import gt_main
#plot target coordinates on images
for i in range(2060,2070):
    image_id=i
    g=target[image_id,:,:,:]
    ig=np.zeros([7,7,73])   # 7x7x(10+num_classes)
    ig[:,:,0:63]=g[:,:,0:63]   # , 0: num_classes
    ig[:,:,64]=g[:,:,67]  #  num_calsses+1]][ target_last one
    ig[:,:,69:73]=g[:,:,63:67]  # predict_last four]][ num_calsses: num_classes +4
    ig=np.expand_dims(ig,axis=0)
    img=images[image_id,:,:,:].copy()
    a=gt_main(ig,img,i)

    





'''import os.path
save_path = '/home/user01/data_ssd/Abbas/OBJECT_DETECTION_YOLO/synthetic_Fruit/groundtruths/'
for i in range(4):
    name=str(i)
    clas="banana"
    Xmin=15
    Ymin=10
    Xmax=20
    Ymax=100
    file = os.path.join(save_path, name +".txt")  
    file = open(file, "w") 
    file.write(clas + ' ') 
    file.write(str(Xmin) + ' ') 
    file.write(str(Ymin ) + ' ') 
    file.write(str(Xmax ) + ' ') 
    file.write(str(Ymax )+ '\n') 
    file.write(clas + ' ') 
    file.write(str(Xmin) + ' ') 
    file.write(str(Ymin ) + ' ') 
    file.write(str(Xmax ) + ' ') 
    file.write(str(Ymax ) + ' ') 
    file.close()'''
    
    

h=0.54
print(str(h))


