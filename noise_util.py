
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import math


def cal_dis(pA, pB):
    return math.sqrt((pA[0] - pB[0])**2 + (pA[1] - pB[1])**2)


def add_noise(img, generate_ratio=0.001, generate_size=0.003):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R_max = max(3, int(min(h, w) * generate_size))
    for _ in range(int(h * w * generate_ratio)):
        R = random.randint(1, R_max)
        P_noise_x = random.randint(R, w - 1 - R)
        P_noise_y = random.randint(R, h - 1 - R)
        for i in range(P_noise_x - R, P_noise_x + R):
            for j in range(P_noise_y - R, P_noise_y + R):
                if cal_dis((i, j), (P_noise_x, P_noise_y)) < R:
                    if random.random() < 0.8:
                        if random.random() < 0.5:
                            img[j][i] = 0
                        else:
                            img[j][i] = 255
        R = random.randint(1, R_max * 2)
        P_noise_x = random.randint(0, w - 1 - R)
        P_noise_y = random.randint(0, h - 1 - R)
        for i in range(P_noise_x + 1, P_noise_x + R):
            for j in range(P_noise_y + 1, P_noise_y + R):
                if random.random() < 0.8:
                    if random.random() < 0.5:
                        img[j][i] = 0
                    else:
                        img[j][i] = 255
    img = Image.fromarray(img)
    return img


if __name__ == '__main__':
    img = Image.open('data/book_pages/imgs_vertical/book_page_1.jpg')
    img = add_noise(img)
    img.show()
