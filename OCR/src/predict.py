import os
import time
import torch
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn_v2
import argparse
from model import CRNN
from utils import predict


max_object_line = 50
num_classes_line = 3
num_classes_char = 6000
line_size = (384, 24)

nc = 1
nh = 384
nclass = num_classes_char + 1
height = 24
layout_lambda = 0.25

output_raw = True
output_text = False
output_positions = False
output_image = False
line_break = False
overwrite = False
output_confidence = False

character_source = "data/source/CCL6kT.txt"
model_path_det = 'models/frcnnv2_det_50_mixed.pth'
model_path_rec = 'models/crnn_rec_6000_mixed.pth'

input_path = "target/images/"
output_path = "output/"
input_path_test = "target/samples/images/"
output_path_test = "output/samples/"


"""
Example usage:
    python predict.py -t
This command will use sample test images. Without the -t flag, the script will use the images in the target/images directory.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='Use test images')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('-c', '--confidence', action='store_true', help='Save confidence')
    args = parser.parse_args()

    if args.test:
        input_path = input_path_test
        output_path = output_path_test
    if args.overwrite:
        overwrite = True
    if args.confidence:
        output_confidence = True
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device}')

    character_list = []
    with open(character_source, "r", encoding='UTF-8') as file:
        # Read all characters at once
        characters = file.read()
        # Convert the string to a list of characters
        character_list = list(characters)

    # Load pre-trained Faster R-CNN model for line detection
    model_line = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', box_detections_per_img=max_object_line)
    in_features = model_line.roi_heads.box_predictor.cls_score.in_features
    model_line.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes_line)
    state_dict = torch.load(model_path_det) if torch.cuda.is_available() else torch.load(model_path_det, map_location=torch.device('cpu'))
    model_line.load_state_dict(state_dict)
    model_line = model_line.eval().to(device)
    print("Load detection model:", model_path_det)

    # Load pre-trained CRNN model for text recognition
    model_char = CRNN(nc, nh, nclass, height)
    state_dict = torch.load(model_path_rec) if torch.cuda.is_available() else torch.load(model_path_rec, map_location=torch.device('cpu'))
    model_char.load_state_dict(state_dict)
    model_char.eval().to(device)
    print("Load recognition model:", model_path_rec)

    file_list = os.listdir(input_path)
    file_list.sort()
    start_time = time.time()

    for filename in file_list:
        predict(filename, input_path, output_path, model_line, model_char, device, character_list, num_classes_char, line_size, output_raw=output_raw, output_text=output_text, output_positions=output_positions, output_image=output_image, output_confidence=output_confidence, line_break=line_break, overwrite=overwrite, layout_lambda=layout_lambda)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s, Average time per file: {(end_time - start_time) / len(os.listdir(input_path)):.2f}s")