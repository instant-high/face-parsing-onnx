import cv2
import numpy as np
import onnxruntime
from argparse import ArgumentParser

#
from face_parser.face_parser import FACE_PARSER
facemask = FACE_PARSER(model_path="face_parser/face_parser.onnx", device="cuda")
#

parser = ArgumentParser()
parser.add_argument("--source_image", default='image.jpg', help="path to source image")
parser.add_argument("--parser_index", default="1, 2, 3, 4, 5, 6, 10, 11, 12, 13", type=lambda x: list(map(int, x.split(','))),help='index of mask parts')
opt = parser.parse_args()

part_index = opt.parser_index
assert type(part_index) == list

face = cv2.imread(opt.source_image)
face = cv2.resize(face,(512,512))

#
mask = facemask.create_region_mask(face, part_index)
#
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

cv2.imshow("Result parts: " + str(part_index), mask)
cv2.waitKey()