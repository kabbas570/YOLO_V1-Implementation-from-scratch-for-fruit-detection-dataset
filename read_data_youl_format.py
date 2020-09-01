import numpy as np
import cv2
import matplotlib.pyplot as plt




def data_readV():   
    image_w=448
    image_h=448
    img_ = []
    tar_=[]    
    with open('_annotationsV.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    for i in range(len(content)):
        label_matrix = np.zeros([7, 7, 68])  # 7 x 7 x num_classes + 5
        img_path=content[i].partition(" ")[0]
        img=cv2.imread(img_path)
        img = cv2.resize(img, (image_w,image_h), interpolation = cv2.INTER_AREA) 
        img=img/255
        img_.append(img)
        q=0
        for j in content[i]:
            if j==" ":
                q=q+1
        for c in range(q):
            label=content[i].split(" ")[c+1]      
            for  l  in  label :
                l  = label. split ( ',' )
                l = np.array(l, dtype=np.int)
                xmin = l[0]*(14/13)
                ymin  = l[ 1 ]*(224/275)
                xmax = l[2]*(14/13)
                ymax = l[3]*(224/275)
                clas  =  l [ 4 ]
                x = (xmin + xmax) / 2 / image_w
                y = (ymin + ymax) / 2 / image_h
                w = (xmax - xmin) / image_w
                h = (ymax - ymin) / image_h
                loc = [7 * x, 7 * y]
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                y = loc[1] - loc_i
                x = loc[0] - loc_j          
                label_matrix[loc_i, loc_j, clas] = 1
                label_matrix[loc_i, loc_j, 63:67] = [x, y, w, h]   # , num_classes:num_classes+4
                label_matrix[loc_i, loc_j, 67] = 1  # response      # las index  i.e num_class+4
        tar_.append(label_matrix)
    img_ = np.array(img_)
    tar_ = np.array(tar_)
    return img_,tar_



def data_read():   
    image_w=448
    image_h=448
    img_ = []
    tar_=[]    
    with open('_annotations.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    for i in range(len(content)):
        label_matrix = np.zeros([7, 7, 68])  # 7 x 7 x num_classes + 5
        img_path=content[i].partition(" ")[0]
        img=cv2.imread(img_path)
        img = cv2.resize(img, (image_w,image_h), interpolation = cv2.INTER_AREA) 
        img=img/255
        img_.append(img)
        q=0
        for j in content[i]:
            if j==" ":
                q=q+1
        for c in range(q):
            label=content[i].split(" ")[c+1]      
            for  l  in  label :
                l  = label. split ( ',' )
                l = np.array(l, dtype=np.int)
                xmin = l[0]*(14/13)             #  desired/image.shape[1]
                ymin  = l[ 1 ]*(224/275)   #  desired/image.shape[0]
                xmax = l[2]*(14/13)     #  desired/image.shape[1]
                ymax = l[3]*(224/275)    #  desired/image.shape[0]
                clas  =  l [ 4 ]
                x = (xmin + xmax) / 2 / image_w
                y = (ymin + ymax) / 2 / image_h
                w = (xmax - xmin) / image_w
                h = (ymax - ymin) / image_h
                loc = [7 * x, 7 * y]
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                y = loc[1] - loc_i
                x = loc[0] - loc_j          
                label_matrix[loc_i, loc_j, clas] = 1
                label_matrix[loc_i, loc_j, 63:67] = [x, y, w, h]   # , num_classes:num_classes+4
                label_matrix[loc_i, loc_j, 67] = 1  # response      # las index  i.e num_class+4
        tar_.append(label_matrix)
    img_ = np.array(img_)
    tar_ = np.array(tar_)
    return img_,tar_


ikm=cv2.imread('996_jpg.rf.3c00a2b62a0f27a641a62bdd817ac6b6.jpg')