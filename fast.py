from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

img = Image.open('input/1.png').convert('L')
arr = np.asarray(img)
height, width = arr.shape

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

def has_n_contiguous_pixels(arr: np.array, n: int) -> bool:
    extended_arr = arr + arr
    count = 0
    for i in extended_arr:
        if i:
            count +=1
            if count >= n:
                return True
        else:
            count = 0
    return False

def high_speed_test(arr: np.array, threshold: int, x: int, y: int) -> bool:
    test_pixels = [(0,-3),(0,3),(3,0),(-3,0)]
    brighter_count = 0
    darker_count = 0
    for dx, dy in test_pixels:
        if arr[y+dy, x+dx] > arr[y, x] + threshold:
            brighter_count += 1
        if arr[y+dy, x+dx] < arr[y, x] - threshold:
            darker_count += 1
    if brighter_count >= 2 or darker_count >= 2:
        return True
    else:
        return False

start_time = time.time()
corners = []
for x in range(3, width-3):
    for y in range(3, height-3):
        if not high_speed_test(arr, threshold, x, y):
            continue
        brighter = []
        darker = []
        for dx, dy in circle_offsets:
            brighter.append(arr[y+dy, x+dx] > arr[y, x] + threshold)
            darker.append(arr[y+dy, x+dx] < arr[y, x] - threshold)
        if has_n_contiguous_pixels(brighter, n) or has_n_contiguous_pixels(darker, n):
            corners.append((x,y))
end_time = time.time()
print(f"Execution Time: {(end_time - start_time):.2f}")

img_rgb = img.convert("RGB")
draw = ImageDraw.Draw(img_rgb)

for x, y in corners:
    draw.ellipse((x-1, y-1, x+1, y+1), fill="red")
dpi = 100
figsize = (width / dpi, height / dpi)

plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(img_rgb)
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('./output.png')
plt.show()