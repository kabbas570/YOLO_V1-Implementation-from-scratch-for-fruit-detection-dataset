import numpy as np
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt
import os
path='/home/user01/data_ssd/Abbas/OBJECT_DETECTION_YOLO/synthetic_Fruit/results/'
classes_name = ['Apple','Apricot','Avocado','Banana','Beetroot','Blueberry','Cactus','Cantaloupe','Carambula','Cauliflower','Cherry','Chestnut','Clementine',
                'Cocos','Dates','Eggplant','Ginger','Granadilla','Grape','Grapefruit','Guava'
,'Hazelnut','Huckleberry','Kaki','Kiwi','Kohlrabi','Kumquats','Lemon','Limes','Lychee','Mandarine','Mango','Mangostan','Maracuja','Melon'
,'Mulberry','Nectarine','Nut','Onion','Orange','Papaya','Passion','Peach','Pear','Pepino','Pepper','Physalis','Pineapple','Pitahaya','Plum'
,'Pomegranate','Pomelo','Potato','Quince','Rambutan','Raspberry','Redcurrant','Salak','Strawberry','Tamarillo','Tangelo','Tomato','Walnut']
num_classes=63

import os.path
save_path = '/home/user01/data_ssd/Abbas/OBJECT_DETECTION_YOLO/synthetic_Fruit/detections/'


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max
def yolo_head1(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = np.shape(feats)[0:2]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = np.arange(0, stop=conv_dims[0])
    conv_width_index = np.arange(0, stop=conv_dims[1])
    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = np.reshape(np.transpose(conv_width_index), [conv_dims[0] * conv_dims[1]])
    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    conv_index = np.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])

    conv_dims = np.reshape(conv_dims, [1, 1, 1, 2])

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    return iou_scores

def _main(prediction,image,img_num): #prediction=(1,7,7,11) and image= 416,416,3
    confidences=[]
    name=str(img_num)
    file = os.path.join(save_path, name +".txt")  
    file = open(file, "w") 
    #prediction =model.predict(images[14:15,:,:,:])
    prediction=prediction
    predict_class = prediction[..., :num_classes]  # 1 * 7 * 7 * 20
    predict_trust = prediction[..., num_classes:num_classes+2]  # 1 * 7 * 7 * 2
    predict_box = prediction[..., num_classes+2:]  # 1 * 7 * 7 * 8

    predict_class = np.reshape(predict_class, [7, 7, 1, num_classes])
    predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])
    predict_box = np.reshape(predict_box, [7, 7, 2, 4])

    predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 20

    box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2
    box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2
    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)  # 7 * 7 * 1

    box_mask = box_class_scores >= best_box_class_scores  # ? * 7 * 7 * 2

    filter_mask = box_class_scores >= 0.2  # 7 * 7 * 2
    filter_mask *= box_mask  # 7 * 7 * 2

    filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1

    predict_scores *= filter_mask  # 7 * 7 * 2 * 20
    predict_box *= filter_mask  # 7 * 7 * 2 * 4

    box_classes = np.expand_dims(box_classes, axis=-1)
    box_classes *= filter_mask  # 7 * 7 * 2 * 1

    box_xy, box_wh = yolo_head1(predict_box)  # 7 * 7 * 2 * 2
    box_xy_min, box_xy_max = xywh2minmax(box_xy, box_wh)  # 7 * 7 * 2 * 2

    predict_trust *= filter_mask  # 7 * 7 * 2 * 1
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1
    predict_trust_max = np.max(predict_trust)  
    max_i = max_j = max_k = 0
    while predict_trust_max > 0:
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if predict_trust[i, j, k, 0] == predict_trust_max:
                        nms_mask[i, j, k, 0] = 1
                        filter_mask[i, j, k, 0] = 0
                        max_i = i
                        max_j = j
                        max_k = k
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if filter_mask[i, j, k, 0] == 1:
                        iou_score = iou(box_xy_min[max_i, max_j, max_k, :],
                                        box_xy_max[max_i, max_j, max_k, :],
                                        box_xy_min[i, j, k, :],
                                        box_xy_max[i, j, k, :])
                        if iou_score > 0.7:
                            filter_mask[i, j, k, 0] = 0
        predict_trust *= filter_mask  # 7 * 7 * 2 * 1
        predict_trust_max = np.max(predict_trust)
        confidences.append(predict_trust_max)
    box_xy_min *= nms_mask
    box_xy_max *= nms_mask
    print(confidences)
    image =image 
    origin_shape = image.shape[0:2]
    detect_shape = filter_mask.shape
    q=0
    for i in range(detect_shape[0]):
        for j in range(detect_shape[1]):
            for k in range(detect_shape[2]):
                if nms_mask[i, j, k, 0]:
                    cv2.rectangle(image, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                 (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                                 (1, 1, 1),thickness=2)
                    cv2.putText(image, classes_name[box_classes[i, j, k, 0]],
                               (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                               1, 1, (1, 1, 1),thickness=2)
                    clas=classes_name[box_classes[i, j, k, 0]]
                    Xmin=int(box_xy_min[i, j, k, 0])
                    Ymin=int(box_xy_min[i, j, k, 1])
                    Xmax=int(box_xy_max[i, j, k, 0])
                    Ymax=int(box_xy_max[i, j, k, 1])    
                    print(box_classes[i, j, k, 0])
                    file.write(clas + ' ')                   
                    if box_classes[i, j, k, 0]!=-1:
                        file.write(str(confidences[q]) + ' ')
                        q=q+1
                    file.write(str(Xmin) + ' ') 
                    file.write(str(Ymin ) + ' ') 
                    file.write(str(Xmax ) + ' ') 
                    file.write(str(Ymax )+ '\n')                  
    file.close()                 
    image = cv2.resize(image, (origin_shape[1], origin_shape[0]))
    cv2.imwrite(os.path.join(path , str(img_num)+".png"),image*255) 
    
