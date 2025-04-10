from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

img = Image.open('input/8.png').convert('L')
ori_image = np.asarray(img, dtype=np.uint8)
ori_height, ori_width = ori_image.shape

#            (-1,-3), (0,-3), (1,-3),
#        (-2,-2),                   (2,-2),
#    (-3,-1),                         (3,-1),
#    (-3,0),                           (3,0),
#    (-3,1),                           (3,1),
#        (-2,2),                     (2,2),
#               (-1,3), (0,3), (1,3),

circle_offsets = [
    (0,-3),(1,-3),(2,-2),(3,-1),(3,0),(3,1),(2,2),(1,3),
    (0,3),(-1,3),(-2,2),(-3,1),(-3,0),(-3,-1),(-2,-2),(-1,-3)
]
threshold = 20
n=9

def has_n_contiguous_pixels(check_list: list[bool], n: int) -> bool:
    extended_check_list = check_list + check_list
    count = 0
    for i in extended_check_list:
        if i:
            count +=1
            if count >= n:
                return True
        else:
            count = 0
    return False

def high_speed_test(image: np.ndarray, threshold: int, x: int, y: int) -> bool:
    test_pixels = [(0,-3),(0,3),(3,0),(-3,0)]
    brighter_count = 0
    darker_count = 0
    for dx, dy in test_pixels:
        if image[y+dy, x+dx] > image[y, x] + threshold:
            brighter_count += 1
        if image[y+dy, x+dx] < image[y, x] - threshold:
            darker_count += 1
    if brighter_count >= 3 or darker_count >= 3:
        return True
    else:
        return False

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

def non_max_suppression(corners: list[tuple[int, int]], image: np.ndarray, grid_size: int = 5) -> list[tuple[int, int]]:
    if not corners:
        return []

    grid_map = {}

    for x, y in corners:
        center = int(image[y, x])
        score = sum(abs(int(image[y + dy, x + dx]) - center) for dx, dy in circle_offsets)
        grid_x = x // grid_size
        grid_y = y // grid_size
        grid_key = (grid_y, grid_x)
        if grid_key not in grid_map or score > grid_map[grid_key][0]:
            grid_map[grid_key] = (score, (x, y))

    filtered_corners = [corner for _, corner in grid_map.values()]
    return filtered_corners

# Resize
start_time = time.time()
target_height, target_width = calculate_target_size(ori_height, ori_width, 300)
#target_width, target_height = (ori_width, ori_height)
resized_image = resize_image(ori_image, target_height, target_width)
end_time = time.time()
print(f"Resize Execution Time: {(end_time - start_time):.2f} s")

# Fast Keypoint Detection 
start_time = time.time()
corners = []
for x in range(3, target_width-3):
    for y in range(3, target_height-3):
        if not high_speed_test(resized_image, threshold, x, y):
            continue
        brighter = []
        darker = []
        for dx, dy in circle_offsets:
            brighter.append(resized_image[y+dy, x+dx] > resized_image[y, x] + threshold)
            darker.append(resized_image[y+dy, x+dx] < resized_image[y, x] - threshold)
        if has_n_contiguous_pixels(brighter, n) or has_n_contiguous_pixels(darker, n):
            corners.append((x,y))
end_time = time.time()
print(f"Detection Execution Time: {(end_time - start_time):.2f} s")
print(f"Detected corners before NMS: {len(corners)}")

# Non-Maximum Suppression
start_time = time.time()
filtered_corners = non_max_suppression(corners, resized_image, grid_size=5)
end_time = time.time()
print(f"NMS Execution Time: {(end_time - start_time):.2f} s")
print(f"Detected corners after NMS: {len(filtered_corners)}")

# Mapping and Visualization
start_time = time.time()
mapped_corners = map_keypoints_to_original(corners, ori_height, ori_width, target_height, target_width)

#img = Image.fromarray(resized_image)
img_rgb = img.convert("RGB")
draw = ImageDraw.Draw(img_rgb)

for x, y in mapped_corners:
    draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
dpi = 100
figsize = (ori_width / dpi, ori_height / dpi)

plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(img_rgb)
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
end_time = time.time()
print(f"Mapping and Visualization Execution Time: {(end_time - start_time):.2f} s")
plt.savefig('./output.png')
plt.show()