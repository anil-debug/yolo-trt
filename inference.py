#Kumar#
import cv2
from typing import Tuple, List, Union
import numpy as np
from pathlib import Path
#import pycuda_api
from pycuda_api import TRTEngine
import random 
import math
import glob
import os, sys
# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')
random.seed(0)
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5



def letterbox(im,new_shape = (640, 640),color = (114, 114, 114)): 
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im, return_seg = False):
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images


def crop_mask(masks, bboxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))






class yolov8():
    def __init__(self, engine_path, conf_thres = 0.25, iou_thres = 0.65):
        self.engine = TRTEngine(engine_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.h, self.w = self.engine.inp_info[0].shape[-2:]
        self.dw=None
        self.dh=None
        self.dwdh=None
        self.ratio=None
        self.seg_img=None
        self.tensor=None
        self.new_bgr=None
        self.orig_shape=None

    def preprocess(self, bgr):
        self.orig_shape=bgr
        bgr, self.ratio, self.dwdh = letterbox(bgr, (self.w, self.h))
        self.new_bgr= bgr
        self.dw, self.dh = int(self.dwdh[0]), int(self.dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.tensor, self.seg_img = blob(rgb, return_seg=True)
        self.dwdh = np.array(self.dwdh * 2, dtype=np.float32)
        self.tensor = np.ascontiguousarray(self.tensor)
        self.seg_img=self.seg_img[self.dh:self.h-self.dh,self.dw:self.w-self.dw,[2,1,0]]
        return self.tensor,self.seg_img,self.dh, self.dw, self.ratio,self.dwdh

    def inference(self, tensor):
        tensor = self.tensor
        data = self.engine(tensor)
        print(len(data))
        return data

    def sigmoid(self,x):
        return 1. / (1. + np.exp(-x))    

    def seg_postprocess(self, data, shape):
        assert len(data) == 2
        h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
        outputs, proto = (i[0] for i in data)
        bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
        scores, labels = scores.squeeze(), labels.squeeze()
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        idx = scores > conf_thres
        bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx], maskconf[idx]
        cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], 1)
        labels = labels.astype(np.int32)
        v0, v1 = map(int, (cv2.__version__).split('.')[:2])
        assert v0 == 4, 'OpenCV version is wrong'
        if v1 > 6:
            idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres, iou_thres)
        else:
            idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
        bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx], maskconf[idx]
        masks = self.sigmoid(maskconf @ proto).reshape(-1, h, w)
        masks = crop_mask(masks, bboxes / 4.)
        masks = np.transpose(masks, [1, 2, 0]) # swap height and width dimensions
        # if masks is not None and masks.size != 0 and shape is not None and len(shape) == 2:
        #     print("masks are not none")
        masks = cv2.resize(masks, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        masks = np.transpose(masks, [2, 0, 1]) # swap back to original dimensions
        masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
        #print("in segpostproc",masks)
        return bboxes, scores, labels, masks
    


    def postprocess(self,res, bgr):
        bgr = self.new_bgr
        bboxes, scores, labels, masks = self.seg_postprocess(res, bgr.shape[:2])
        masks = masks[:, self.dh:self.h - self.dh, self.dw:self.w - self.dw, :]
        mask_colors = MASK_COLORS[labels % len(MASK_COLORS)]
        mask_colors = mask_colors.reshape(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (self.seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = self.orig_shape
        draw = cv2.resize(seg_img.astype(np.uint8), draw.shape[:2][::-1])
        bboxes -= self.dwdh
        bboxes /= self.ratio
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        cv2.imshow("result", draw)
        cv2.waitKey(0)
        return draw, bboxes, scores, labels, masks


if __name__ == '__main__':
    #trt_model = 'model_final.engine'
    #trt_model = './yolov8s-seg.engine'
    trt_model = './yolov8s-seg.engine'
    img_folder_path = 'data/zidane.jpg'
 
    yolov8 = yolov8(engine_path= trt_model, conf_thres = 0.25, iou_thres = 0.65)

    
    for img_path in glob.glob(img_folder_path):
        if os.path.isfile(img_path):
            print(f"Image name : {img_path}")
            ori_img = cv2.imread(img_path)
            img = yolov8.preprocess(ori_img)
            # cv2.imshow("window",img)
            print("finished pre-processing")
            res = yolov8.inference(img)
            #print(res.shape)
            # cv2.imshow("window2", res)
            print("finished inference")
            #out = yolov8.postprocess(res, ori_img)
            img,bboxes,scores,labels,masks = yolov8.postprocess(res, ori_img)
            print("finished post-processing")
           
