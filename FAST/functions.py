import numpy as np

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

def non_max_suppression_grid(corners: list[tuple[int, int]], image: np.ndarray, grid_size: int = 5) -> list[tuple[int, int]]:
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

def non_max_suppression_window(corners: list[tuple[int, int]], image: np.ndarray, window_size: int = 5) -> list[tuple[int, int]]:
    if not corners:
        return []

    scores = []
    for x, y in corners:
        center = int(image[y, x])
        score = sum(abs(int(image[y + dy, x + dx]) - center) for dx, dy in circle_offsets)
        scores.append(score)

    half_window = window_size // 2
    filtered_corners = []
    for i, (x, y) in enumerate(corners):
        is_max = True
        for j, (x2, y2) in enumerate(corners):
            if i != j and abs(x - x2) <= half_window and abs(y - y2) <= half_window:
                if scores[j] > scores[i]:
                    is_max = False
                    break
        if is_max:
            filtered_corners.append((x, y))

    return filtered_corners