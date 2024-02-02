# face-parsing-onnx
face parsing / face masking converted to onnx model

CPU or GPU

Requirements: opencv-python, numpy, onnxruntime (onnruntime-gpu, cudatoolkit=11.2 cudnn=8.1.0)

python inference.py --source_image "image.jpg" --parser_index 1,2,3,4,5,6,10,11,12,13


Original torch repository:

https://github.com/zllrunning/face-parsing.PyTorch


