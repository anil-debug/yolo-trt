# Kumar

This script is designed to perform object detection and instance segmentation using a YOLOv8 model. Here's a breakdown of the functionalities and components of the script:

## Requirements
- `cv2`: OpenCV library for image processing.
- `numpy`: Library for numerical computations.
- `pathlib`: Library for handling file paths.
- `pycuda_api`: Custom module for interfacing with the GPU.
- Other standard libraries like `random`, `math`, `glob`, `os`, and `sys`.

## Constants
- `SUFFIXS`: Tuple containing supported image file extensions.
- `CLASSES`: Tuple containing class labels for object detection.
- `COLORS`: Dictionary mapping class labels to random RGB colors.
- `MASK_COLORS`: Array of colors used for segment masks.
- `ALPHA`: Alpha value for segment masks.

## Functions
- `letterbox`: Resizes and pads images while meeting stride-multiple constraints.
- `blob`: Converts image to blob format for inference.
- `sigmoid`: Sigmoid activation function.
- `path_to_list`: Converts image paths to a list.
- `crop_mask`: Crops segment masks based on bounding boxes.
- `yolov8`: Class for YOLOv8 model with methods for preprocessing, inference, and postprocessing.
- `preprocess`: Preprocesses images for inference.
- `inference`: Performs inference on preprocessed images.
- `seg_postprocess`: Postprocesses segmentation outputs.
- `postprocess`: Combines object detection and segmentation outputs for visualization.

## Usage
- Instantiate the `yolov8` class with the path to the TRT engine file.
- Call the `preprocess` method to prepare images for inference.
- Perform inference using the `inference` method.
- Visualize the results using the `postprocess` method.

## Example
```python
trt_model = './yolov8s-seg.engine'
img_folder_path = 'data/zidane.jpg'

yolov8 = yolov8(engine_path=trt_model, conf_thres=0.25, iou_thres=0.65)

for img_path in glob.glob(img_folder_path):
    if os.path.isfile(img_path):
        ori_img = cv2.imread(img_path)
        img = yolov8.preprocess(ori_img)
        res = yolov8.inference(img)
        img, bboxes, scores, labels, masks = yolov8.postprocess(res, ori_img)
