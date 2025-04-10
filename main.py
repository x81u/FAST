from FAST.utils import calculate_target_size, resize_image, map_keypoints_to_original, draw_keypoints_to_image
from FAST.detector import fast_keypoint_detector
from FAST.functions import non_max_suppression

from PIL import Image
import numpy as np
import time
import json

# Load Config
with open('./config.json', 'r') as f:
    data = f.read()
    config = json.loads(data)

image_path = config['image_path']
resize_acceleration = config['resize_acceleration']
threshold = config['threshold']
n = config['n']
max_size = config['max_size']
grid_size = config['grid_size']
draw_radius = config['draw_radius']
file_name = image_path.split('.')[0].split('/')[-1]

if __name__ == '__main__':
    # Read Image
    img = Image.open(image_path).convert('L')
    ori_image = np.asarray(img, dtype=np.uint8)
    ori_height, ori_width = ori_image.shape

    # Resize
    if resize_acceleration:
        target_height, target_width = calculate_target_size(ori_height, ori_width, max_size)
        resized_image = resize_image(ori_image, target_height, target_width)
    else:
        target_height, target_width = ori_height, ori_width
        resized_image = ori_image

    # Fast Keypoint Detection 
    start_time = time.time()
    corners = fast_keypoint_detector(resized_image, target_height, target_width, threshold, n)
    end_time = time.time()
    print(f"Detection Execution Time: {(end_time - start_time):.2f} s")
    print(f"Detected corners before NMS: {len(corners)}")

    # Non-Maximum Suppression
    start_time = time.time()
    filtered_corners = non_max_suppression(corners, resized_image, grid_size)
    end_time = time.time()
    print(f"NMS Execution Time: {(end_time - start_time):.2f} s")
    print(f"Detected corners after NMS: {len(filtered_corners)}")

    # Mapping and Visualization
    mapped_corners = map_keypoints_to_original(filtered_corners, ori_height, ori_width, target_height, target_width)
    img_rgb = img.convert("RGB")
    draw_keypoints_to_image(img_rgb, mapped_corners, ori_width, ori_height, draw_radius, file_name)