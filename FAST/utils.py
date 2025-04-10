import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw

def calculate_target_size(ori_height: int, ori_width: int, max_size: int) -> tuple[int, int]:
    aspect_ratio = ori_width / ori_height
    if ori_height > ori_width:
        target_height = max_size
        target_width = int(max_size * aspect_ratio)
    else:
        target_width = max_size
        target_height = int(max_size / aspect_ratio)
    return target_height, target_width

def resize_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    ori_height, ori_width = image.shape
    resized = np.zeros((target_height, target_width), dtype=np.uint8)
    
    y_ratio = ori_height / target_height
    x_ratio = ori_width / target_width
    
    for y in range(target_height):
        for x in range(target_width):
            orig_y = int(y * y_ratio)
            orig_x = int(x * x_ratio)
            resized[y, x] = image[orig_y, orig_x]
    
    return resized

def map_keypoints_to_original(keypoints: list[int], ori_height: int, ori_width: int, resized_height: int, resized_width: int) -> list[int]:
    y_ratio = ori_height / resized_height
    x_ratio = ori_width / resized_width
    mapped_keypoints = [(int(x * x_ratio), int(y * y_ratio)) for x, y in keypoints]
    return mapped_keypoints

def draw_keypoints_to_image(image: Image.Image, corners: list[tuple[int, int]], width: int, height: int, draw_radius: int, file_name: str) -> None:
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for x, y in corners:
        draw.ellipse((x-draw_radius, y-draw_radius, x+draw_radius, y+draw_radius), fill="red")
    dpi = 100
    figsize = (width / dpi, height / dpi)

    os.makedirs('output', exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join('output', file_name+'.png'))
    plt.show()