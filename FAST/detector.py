import numpy as np
from .functions import high_speed_test, has_n_contiguous_pixels, circle_offsets

def fast_keypoint_detector(image: np.ndarray, height: int, width: int, threshold: int, n: int) -> list[tuple[int, int]]:
    corners = []
    for x in range(3, width-3):
        for y in range(3, height-3):
            if not high_speed_test(image, threshold, x, y):
                continue
            brighter = []
            darker = []
            for dx, dy in circle_offsets:
                brighter.append(image[y+dy, x+dx] > image[y, x] + threshold)
                darker.append(image[y+dy, x+dx] < image[y, x] - threshold)
            if has_n_contiguous_pixels(brighter, n) or has_n_contiguous_pixels(darker, n):
                corners.append((x,y))
    return corners