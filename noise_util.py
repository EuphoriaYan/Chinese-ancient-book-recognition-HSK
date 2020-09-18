
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import random
import math
import json


def cal_dis(pA, pB):
    return math.sqrt((pA[0] - pB[0])**2 + (pA[1] - pB[1])**2)


def add_noise(img, generate_ratio=0.003, generate_size=0.004):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R_max = max(3, int(min(h, w) * generate_size))
    threshold = int(h * w * generate_ratio)
    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))
    cnt = 0

    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(R, w - 1 - R)
        P_noise_y = random.randint(R, h - 1 - R)
        for i in range(P_noise_x - R, P_noise_x + R):
            for j in range(P_noise_y - R, P_noise_y + R):
                if cal_dis((i, j), (P_noise_x, P_noise_y)) < R:
                    if random.random() < 0.8:
                        img[j][i] = 0 if random.random() < 0.5 else 255
        cnt += 2 * R
        if cnt >= threshold:
            break

    R_max *= 2
    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))
    cnt = 0
    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(0, w - 1 - R)
        P_noise_y = random.randint(0, h - 1 - R)
        for i in range(P_noise_x + 1, P_noise_x + R):
            for j in range(P_noise_y + 1, P_noise_y + R):
                if random.random() < 0.8:
                    img[j][i] = 0 if random.random() < 0.5 else 255
        cnt += R
        if cnt >= threshold:
            break

    img = Image.fromarray(img)
    return img


if __name__ == '__main__':
    img = Image.open('data/book_pages/imgs_vertical/book_page_0.jpg').convert('RGB')
    draw = ImageDraw.Draw(img)
    with open('data/book_pages/book_pages_tags_vertical_3.txt', 'r', encoding='utf-8') as fp:
        line = fp.readline().strip()
    json_str = line.split('\t')[-1]
    data = json.loads(json_str)
    for text_bbox, text in zip(data['text_bbox_list'], data['text_list']):
        print(text)
        draw.rectangle(text_bbox, outline=(255, 0, 0))
        img.show()
