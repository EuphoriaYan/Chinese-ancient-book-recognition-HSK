# -*- encoding: utf-8 -*-
# Author: euphoria

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from pprint import pprint
import json
import random
import cv2
import re
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from queue import Queue
from torchvision import transforms
from torch import nn
import copy

from config import FONT_FILE_DIR, EXTERNEL_IMAGES_DIR, MAX_ROTATE_ANGLE
from config import BOOK_PAGE_SHAPE_LIST
from config import SHUFA_FILE_DIR
from util import CHAR2ID_DICT, IGNORABLE_CHARS, IMPORTANT_CHARS
from config import BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
from config import BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V

from util import check_or_makedirs
from data_generator.img_utils import rotate_PIL_image
from data_generator.img_utils import find_min_bound_box
from data_generator.img_utils import adjust_img_and_put_into_background
from data_generator.img_utils import reverse_image_color
from noise_util import add_noise
from ocrodeg import *
from generate_font_sample import create_mix_ch_handle


def check_text_type(text_type):
    if text_type.lower() in ("h", "horizontal"):
        text_type = "h"
    elif text_type.lower() in ("v", "vertical"):
        text_type = "v"
    else:
        ValueError("Optional text_types: 'h', 'horizontal', 'v', 'vertical'.")
    return text_type


class generate_text_lines_with_text_handle:
    def __init__(self, obj_num, shape=None, text_type="horizontal", text='野火烧不尽春风吹又生', char_size=64,
                 augment=True, fonts_json='/disks/sdb/projs/AncientBooks/data/chars/font_missing.json', fonts_root=None,
                 bad_font_file='charset/songhei_error_font.txt', experiment_dir='songhei_experiment/',
                 type_fonts='type/宋黑类字符集.txt', embedding_num=250, resume=70000, charset='charset/charset_xl.txt',
                 init_num=0, special_type='normal', segment_type='normal'):
        self.text = Queue()
        for char in text:
            self.text.put(char)
        self.text_type = text_type
        if shape is None:
            self.shape = BOOK_PAGE_SHAPE_LIST
        elif isinstance(shape, tuple) and len(shape) == 2:
            self.shape = shape
        else:
            raise ValueError('shape should be tuple and have length 2')
        self.obj_num = obj_num
        self.char_size = char_size
        self.augment = augment
        self.generate_font_handle = create_mix_ch_handle(
            fonts_json=fonts_json,
            fonts_root=fonts_root,
            bad_font_file=bad_font_file,
            experiment_dir=experiment_dir,
            type_fonts=type_fonts,
            embedding_num=embedding_num,
            resume=resume
        )
        self.init_num = init_num
        self.special_type = special_type
        self.segment_type = segment_type
        self.charset = set([s.strip() for s in open(charset, 'r', encoding='utf-8')])

    def generate_book_page_with_text(self):
        text_type = check_text_type(self.text_type)

        if text_type == "h":
            book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
        elif text_type == "v":
            book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
        else:
            raise ValueError('text_type should be horizontal or vertical')

        check_or_makedirs(book_page_imgs_dir)

        with open(book_page_tags_file, "w", encoding="utf-8") as fw:
            for i in range(self.init_num, self.obj_num):
                if isinstance(self.shape, list):
                    shape = random.choice(self.shape)
                else:
                    shape = self.shape

                self.generate_font_handle.set_cur_font()

                PIL_page, text_bbox_list, text_list, char_bbox_list, char_list = self.create_book_page_with_text(
                    shape,
                    text_type=text_type
                )

                if self.augment:
                    new_text_bbox_list = []
                    new_char_bbox_list = []
                    if random.random() > 0.5:
                        w, h = PIL_page.size
                        # resize_ratio_range = (0.8, 1.3)
                        resize_ratio_range = 1.11
                        resize_ratio = random.uniform(1, resize_ratio_range)
                        # resize_ratio = np.exp(resize_ratio)
                        new_w = int(w / resize_ratio)
                        new_h = int(h * resize_ratio)
                        PIL_page = PIL_page.resize((new_w, new_h))
                        for text_bbox in text_bbox_list:
                            new_text_bbox = (
                                int(text_bbox[0] / resize_ratio),
                                int(text_bbox[1] * resize_ratio),
                                int(text_bbox[2] / resize_ratio),
                                int(text_bbox[3] * resize_ratio),
                            )
                            new_text_bbox_list.append(new_text_bbox)
                        for char_bbox in char_bbox_list:
                            new_char_bbox = (
                                int(char_bbox[0] / resize_ratio),
                                int(char_bbox[1] * resize_ratio),
                                int(char_bbox[2] / resize_ratio),
                                int(char_bbox[3] * resize_ratio),
                            )
                            new_char_bbox_list.append(new_char_bbox)
                        text_bbox_list = new_text_bbox_list
                        char_bbox_list = new_char_bbox_list
                    PIL_page = add_noise(PIL_page)
                    PIL_page = ocrodeg_augment(PIL_page)

                image_tags = {"text_bbox_list": text_bbox_list, "text_list": text_list,
                              "char_bbox_list": char_bbox_list, "char_list": char_list}

                img_name = "book_page_%d.jpg" % i
                save_path = os.path.join(book_page_imgs_dir, img_name)
                PIL_page.save(save_path, format="jpeg")
                fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")

                if i % 50 == 0:
                    print(" %d / %d Done" % (i, self.obj_num))
                    sys.stdout.flush()

    def create_book_page_with_text(self, shape, text_type):
        text_type = check_text_type(text_type)

        # 黑色背景书页
        np_page = np.zeros(shape=shape, dtype=np.uint8)
        page_height, page_width = shape

        # 随机确定是否画边框线及行线
        draw = None
        if random.random() < 0.7:
            PIL_page = Image.fromarray(np_page)
            draw = ImageDraw.Draw(PIL_page)

        # 随机确定书页边框
        margin_w = round(random.uniform(0.01, 0.05) * page_width)
        margin_h = round(random.uniform(0.01, 0.05) * page_height)
        margin_line_thickness = random.randint(2, 6)
        line_thickness = round(margin_line_thickness / 2)
        if draw is not None:
            # 点的坐标格式为(x, y)，不是(y, x)
            draw.rectangle([(margin_w, margin_h), (page_width - 1 - margin_w, page_height - 1 - margin_h)],
                           fill=None, outline="white", width=margin_line_thickness)

        # 记录下文本行的bounding-box
        text_bbox_records_list = []
        text_records_list = []

        # 记录下单字的boundding-box
        char_bbox_records_list = []
        char_records_list = []

        if text_type == "h":  # 横向排列

            # 随机确定文本的行数
            rows_num = random.randint(6, 10)
            row_h = (page_height - 2 * margin_h) / rows_num

            # y-coordinate划分行
            ys = [margin_h + round(i * row_h) for i in range(rows_num)] + [page_height - 1 - margin_h]

            # 画行线，第一条线和最后一条线是边框线，不需要画
            if draw is not None:
                for y in ys[1:-1]:
                    draw.line([(margin_w, y), (page_width - 1 - margin_w, y)], fill="white", width=line_thickness)
                np_page = np.array(PIL_page, dtype=np.uint8)

            # 随机决定字符间距占行距的比例
            char_spacing = (random.uniform(0.02, 0.15), random.uniform(0.0, 0.2))  # (高方向, 宽方向)

            # 逐行生成汉字
            for i in range(len(ys) - 1):
                y1, y2 = ys[i] + 1, ys[i + 1] - 1
                x = margin_w + int(random.uniform(0.0, 1) * margin_line_thickness)
                row_length = page_width - x - margin_w
                _, text_bbox_list, text_list, char_bbox_list, char_list = self.generate_mix_rows_chars_with_text(
                    x, y1, y2, row_length, np_page, char_spacing
                )
                text_bbox_records_list.extend(text_bbox_list)
                text_records_list.extend(text_list)
                char_bbox_records_list.extend(char_bbox_list)
                char_records_list.extend(char_list)

        else:  # 纵向排列

            # 随机决定文本的列数
            # cols_num = random.randint(6, 10)
            # cols_num = random.randint(18, 23)
            # cols_num = random.randint(14, 19)
            cols_num = random.randint(int(page_width / page_height * 8), int(page_width / page_height * 13))
            col_w = (page_width - 2 * margin_w) / cols_num

            # x-coordinate划分列
            xs = [margin_w + round(i * col_w) for i in range(cols_num)] + [page_width - 1 - margin_w, ]

            # 画列线，第一条线和最后一条线是边缘线，不需要画
            if draw is not None:
                for x in xs[1:-1]:
                    draw.line([(x, margin_h), (x, page_height - 1 - margin_h)], fill="white", width=line_thickness)
                np_page = np.array(PIL_page, dtype=np.uint8)

            # 随机决定字符间距占列距的比例
            if self.segment_type == 'normal':
                char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif self.segment_type == 'crowded':
                char_spacing = (random.uniform(-0.1, 0.1), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif self.segment_type == 'spacious':
                char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif self.segment_type == 'mixed':
                rand_num = random.random()
                if rand_num > 0.5:  # 50% crowded
                    char_spacing = (random.uniform(-0.1, 0.1), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                elif rand_num < 0.2:  # 20% spacious
                    char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                else:  # 30% normal
                    char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)

            # 逐列生成汉字，最右边为第一列
            for i in range(len(xs) - 1, 0, -1):
                x1, x2 = xs[i - 1] + 1, xs[i] - 1
                y = margin_h + int(random.uniform(0.0, 1) * margin_line_thickness)
                col_length = page_height - y - margin_h
                _, text_bbox_list, text_list, char_bbox_list, char_list = self.generate_mix_cols_chars_with_text(
                    x1, x2, y, col_length, np_page, char_spacing
                )
                text_bbox_records_list.extend(text_bbox_list)
                text_records_list.extend(text_list)
                char_bbox_records_list.extend(char_bbox_list)
                char_records_list.extend(char_list)

        # 将黑底白字转换为白底黑字
        np_page = reverse_image_color(np_img=np_page)
        PIL_page = Image.fromarray(np_page)

        # print(text_bbox_list)
        # print(len(text_bbox_list))
        # PIL_page.show()

        return PIL_page, text_bbox_records_list, text_records_list, char_bbox_records_list, char_records_list

    def generate_mix_rows_chars_with_text(self, x, y1, y2, row_length, np_background, char_spacing):
        row_height = y2 - y1 + 1
        x_start = x

        text_bbox_list = []
        text_list = []
        char_bbox_list = []
        char_list = []
        flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
        remaining_len = row_length
        while remaining_len >= row_height:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(row_height, remaining_len)
            flag += 1
            if flag % 2 == 1:
                x, text_bbox, text, char_bbox = self.generate_one_row_chars_with_text(
                    x, y1, y2, length, np_background, char_spacing
                )
                text_bbox_list.append(text_bbox)
                text_list.append(text)
                char_bbox_list.extend(char_bbox)
                char_list.extend(text)
            else:
                if self.special_type == 'split' or self.special_type == 'split_num_end':
                    x += length
                else:
                    x, text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2 = self.generate_two_rows_chars_with_text(
                        x, y1, y2, length, np_background, char_spacing
                    )
                    text_bbox_list.append(text1_bbox)
                    text_list.append(text1)
                    text_bbox_list.append(text2_bbox)
                    text_list.append(text2)
                    char_bbox_list.extend(char_bbox1)
                    char_list.extend(text1)
                    char_bbox_list.extend(char_bbox2)
                    char_list.extend(text2)
            remaining_len = row_length - (x - x_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        return x, text_bbox_list, text_list, char_bbox_list, char_list

    def generate_mix_cols_chars_with_text(self, x1, x2, y, col_length, np_background, char_spacing):
        col_width = x2 - x1 + 1
        y_start = y

        text_bbox_list = []
        text_list = []
        char_bbox_list = []
        char_list = []
        flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
        remaining_len = col_length
        while remaining_len >= col_width:
            flag += 1
            if flag % 2 == 1:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(col_width, min(remaining_len, col_width * 20))
                y, text_bbox, text, char_bbox = self.generate_one_col_chars_with_text(
                    x1, x2, y, length, np_background, char_spacing
                )
                text_bbox_list.append(text_bbox)
                text_list.append(text)
                char_bbox_list.extend(char_bbox)
                char_list.extend(text)
            else:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(col_width, min(remaining_len, col_width * 10))
                if self.special_type == 'split' or self.special_type == 'split_num_end':
                    y += length
                else:
                    y, text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2 = self.generate_two_cols_chars_with_text(
                        x1, x2, y, length, np_background, char_spacing
                    )
                    text_bbox_list.append(text1_bbox)
                    text_list.append(text1)
                    text_bbox_list.append(text2_bbox)
                    text_list.append(text2)
                    char_bbox_list.extend(char_bbox1)
                    char_list.extend(text1)
                    char_bbox_list.extend(char_bbox2)
                    char_list.extend(text2)
            remaining_len = col_length - (y - y_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        return y, text_bbox_list, text_list, char_bbox_list, char_list

    def generate_one_row_chars_with_text(self, x, y1, y2, length, np_background, char_spacing):
        """
        :return: x, text_bbox, text
        """
        # 记录下生成的汉字及其bounding-box
        char_and_box_list = []

        row_end = x + length - 1
        row_height = y2 - y1 + 1
        while length >= row_height:
            if length < 1.5 * row_height:
                last_char = True
            else:
                last_char = False
            chinese_char, bounding_box, x_tail = self.generate_char_img_into_unclosed_box_with_text(
                np_background, x1=x, y1=y1, x2=None, y2=y2, char_spacing=char_spacing, last_char=last_char
            )

            char_and_box_list.append((chinese_char, bounding_box))
            added_length = x_tail - x
            length -= added_length
            x = x_tail

        # 获取文本行的bounding-box
        head_x1, head_y1, _, _ = char_and_box_list[0][1]
        _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
        text_bbox = (head_x1, head_y1, tail_x2, tail_y2)
        text = [char_and_box[0] for char_and_box in char_and_box_list]
        char_bbox = [char_and_box[1] for char_and_box in char_and_box_list]

        return x, text_bbox, text, char_bbox

    def generate_two_rows_chars_with_text(self, x, y1, y2, length, np_background, char_spacing):
        row_height = y2 - y1 + 1
        mid_y = y1 + round(row_height / 2)

        # x, text_bbox, text
        x_1, text1_bbox, text1, char_bbox1 = self.generate_one_row_chars_with_text(
            x, y1, mid_y, length, np_background, char_spacing)
        x_2, text2_bbox, text2, char_bbox2 = self.generate_one_row_chars_with_text(
            x, mid_y + 1, y2, length, np_background, char_spacing)

        return max(x_1, x_2), text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2

    def generate_one_col_chars_with_text(self, x1, x2, y, length, np_background, char_spacing):
        # 记录下生成的汉字及其bounding-box
        char_and_box_list = []

        col_end = y + length - 1
        col_width = x2 - x1 + 1
        first_char = True
        while length >= col_width:
            # 就算字超过字框了，也不能让它超到页面外面去
            if not first_char:
                if y + (char_spacing[0] + 1) * col_width >= np_background.shape[0]:
                    break
            if length < 1.5 * col_width:
                last_char = True
            else:
                last_char = False
            chinese_char, bounding_box, y_tail = self.generate_char_img_into_unclosed_box_with_text(
                np_background, x1=x1, y1=y, x2=x2, y2=None, char_spacing=char_spacing, first_char=first_char, last_char=last_char
            )
            if chinese_char is None:
                break
            char_and_box_list.append((chinese_char, bounding_box))
            added_length = y_tail - y
            length -= added_length
            y = y_tail
            first_char = False

        # 获取文本行的bounding-box
        head_x1, head_y1, _, _ = char_and_box_list[0][1]
        _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
        text_bbox = (head_x1, head_y1, tail_x2, tail_y2)
        text = [char_and_box[0] for char_and_box in char_and_box_list]
        char_bbox = [char_and_box[1] for char_and_box in char_and_box_list]

        return y, text_bbox, text, char_bbox

    def generate_two_cols_chars_with_text(self, x1, x2, y, length, np_background, char_spacing):
        col_width = x2 - x1 + 1
        mid_x = x1 + round(col_width / 2)

        y_1, text1_bbox, text1, char_bbox1 = self.generate_one_col_chars_with_text(
            mid_x + 1, x2, y, length, np_background, char_spacing)
        y_2, text2_bbox, text2, char_bbox2 = self.generate_one_col_chars_with_text(
            x1, mid_x, y, length, np_background, char_spacing)

        return max(y_1, y_2), text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2

    def generate_char_img_into_unclosed_box_with_text(self, np_background, x1, y1, x2=None, y2=None,
                                                      char_spacing=(0.05, 0.05), first_char=False, last_char=False):

        if x2 is None and y2 is None:
            raise ValueError("There is one and only one None in (x2, y2).")
        if x2 is not None and y2 is not None:
            raise ValueError("There is one and only one None in (x2, y2).")

        chinese_char = ' '
        PIL_char_img = None
        while PIL_char_img is None:
            if (self.special_type == 'num_end' or self.special_type == 'split_num_end') and last_char:
                chinese_char = random.choice(['一', '二', '三'])
            else:
                # 生成白底黑字的字，包含文字
                if not self.text.empty():
                    chinese_char = self.text.get()
                    while chinese_char not in self.charset:
                        chinese_char = self.text.get()
                else:
                    chinese_char = ' '
            PIL_char_img, flag = self.generate_font_handle.get_mix_character(chinese_char)

        PIL_char_img = PIL_char_img.resize((self.char_size, self.char_size))

        # 随机决定是否对汉字图片进行旋转，以及旋转的角度
        if random.random() < 0.35:
            PIL_char_img = rotate_PIL_image(
                PIL_char_img,
                rotate_angle=random.randint(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE))

        # 转为numpy格式
        np_char_img = np.array(PIL_char_img, dtype=np.uint8)

        if chinese_char in IMPORTANT_CHARS or chinese_char == ' ':
            pass
        else:
            # 查找字体的最小包含矩形
            left, right, top, low = find_min_bound_box(np_char_img)
            np_char_img = np_char_img[top:low + 1, left:right + 1]

        char_img_height, char_img_width = np_char_img.shape[:2]

        if x2 is None:  # 文本横向排列
            row_h = y2 - y1 + 1
            char_spacing_h = round(row_h * char_spacing[0])
            char_spacing_w = round(row_h * char_spacing[1])
            if first_char:
                box_x1 = x1
            else:
                box_x1 = x1 + char_spacing_w
            box_y1 = y1 + char_spacing_h
            box_y2 = y2 - char_spacing_h
            box_h = box_y2 - box_y1 + 1

            if char_img_height * 1.4 < char_img_width:
                # 对于“一”这种高度很小、宽度很大的字，应该生成正方形的字图片
                box_w = box_h
                np_char_img = adjust_img_and_put_into_background(np_char_img, background_size=box_h)
            else:
                # 对于宽高相差不大的字，高度撑满，宽度随意
                box_w = round(char_img_width * box_h / char_img_height)
                np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
            box_x2 = box_x1 + box_w - 1

        else:  # y2 is None, 文本纵向排列
            col_w = x2 - x1 + 1
            char_spacing_h = round(col_w * char_spacing[0])
            char_spacing_w = round(col_w * char_spacing[1])
            box_x1 = x1 + char_spacing_w
            box_x2 = x2 - char_spacing_w
            if first_char:
                box_y1 = y1
            else:
                box_y1 = y1 + char_spacing_h
            box_w = box_x2 - box_x1 + 1

            if char_img_width * 1.4 < char_img_height:
                # 对于“卜”这种高度很大、宽度很小的字，应该生成正方形的字图片
                box_h = box_w
                np_char_img = adjust_img_and_put_into_background(np_char_img, background_size=box_w)
            else:
                # 对于宽高相差不大的字，宽度撑满，高度随意
                box_h = round(char_img_height * box_w / char_img_width)
                np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
            box_y2 = box_y1 + box_h - 1

        # 将生成的汉字图片放入背景图片
        try:
            # use 'or' function to make crowded imgs.
            np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] |= np_char_img
        except ValueError as e:
            print('Exception:', e)
            print("The size of char_img is larger than the length of (y1, x1) to edge. Now, resize char_img ...")
            if x2 is None:
                box_x2 = np_background.shape[1] - 1
                box_w = box_x2 - box_x1 + 1
            else:
                box_y2 = np_background.shape[0] - 1
                box_h = box_y2 - box_y1 + 1
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
            try:
                np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] |= np_char_img
            except ValueError as e:
                print('Exception:', e)
                print('Can\'t resize, ignore this char')
                return None, None, None

        # 包围汉字的最小box作为bounding-box
        # bounding_box = (box_x1, box_y1, box_x2, box_y2)

        # 随机选定汉字图片的bounding-box
        bbox_x1 = random.randint(x1, box_x1)
        if box_y1 > y1:
            bbox_y1 = random.randint(y1, box_y1)
        else:
            bbox_y1 = random.randint(box_y1, y1)
        bbox_x2 = min(random.randint(box_x2, box_x2 + char_spacing_w), np_background.shape[1] - 1)
        if char_spacing_h >= 0:
            bbox_y2 = min(random.randint(box_y2, box_y2 + char_spacing_h), np_background.shape[0] - 1)
        else:
            bbox_y2 = min(random.randint(box_y2 + char_spacing_h, box_y2), np_background.shape[0] - 1)
        bounding_box = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

        char_box_tail = box_x2 + 1 if x2 is None else box_y2 + 1

        return chinese_char, bounding_box, char_box_tail


'''
def draw_single_char(ch, font, canvas_size):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return img
    bbox = img.getbbox()
    if bbox is None:
        return img
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 0
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img
'''


# 对字体图像做等比例缩放
def resize_img_by_opencv(np_img, obj_size):
    cur_height, cur_width = np_img.shape[:2]
    obj_width, obj_height = obj_size

    # cv.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # dsize为目标图像大小，(fx, fy)为(横, 纵)方向的缩放比例，参数dsize和参数(fx, fy)不必同时传值
    # interpolation为插值方法，共有5种：INTER_NEAREST 最近邻插值法，INTER_LINEAR 双线性插值法(默认)，
    # INTER_AREA 基于局部像素的重采样，INTER_CUBIC 基于4x4像素邻域的3次插值法，INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值
    # 如果是缩小图片，效果最好的是INTER_AREA；如果是放大图片，效果最好的是INTER_CUBIC(slow)或INTER_LINEAR(faster but still looks OK)
    if obj_height == cur_height and obj_width == cur_width:
        return np_img
    elif obj_height + obj_width < cur_height + cur_width:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    resized_np_img = cv2.resize(np_img, dsize=(obj_width, obj_height), interpolation=interpolation)

    return resized_np_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_num', type=int, required=True)
    parser.add_argument('--text_type', type=str, default='vertical')
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--char_size', type=int, default=64)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--fonts_json', type=str, required=True)
    parser.add_argument('--fonts_root', type=str, default=None)
    parser.add_argument('--bad_font_file', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--type_fonts', type=str, required=True)
    parser.add_argument('--embedding_num', type=int, required=True)
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--init_num', type=int, default=0)
    parser.add_argument('--special_type', type=str, default='normal',
                        choices=['normal', 'split', 'num_end', 'split_num_end'])
    parser.add_argument('--segment_type', type=str, default='normal',
                        choices=['normal', 'crowded', 'spacious', 'mixed'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # with open(r'..\corpus\cleaned\leishu.txt', 'r', encoding='utf-8') as fp:
    #    text = [line.strip() for line in fp]
    #    text = [re.sub('[，。“”‘’？！《》、（）:：；;·［］【】〈〉]', '', line) for line in text]
    #    text = list(filter(None, text))
    args = parse_args()
    pprint(vars(args))
    if args.text_file == 'cover_charset':
        text = []
        with open('./charset/charset_xl.txt', 'r', encoding='utf-8') as fp:
            raw_charset = [line.strip() for line in fp]
        for _ in range(50):
            charset = copy.deepcopy(raw_charset)
            random.shuffle(charset)
            text.extend(charset)
        text = ''.join(text)
    else:
        with open(args.text_file, 'r', encoding='utf-8') as fp:
            text = [line.strip() for line in fp]
            text = [re.sub('[，。“”‘’？！《》、（）〔〕:：；;·［］【】〈〉<>︻︼︵︶︹︺△　]', '', line) for line in text]
            text = list(filter(None, text))
        text = ''.join(text)

    handle = generate_text_lines_with_text_handle(
        obj_num=args.obj_num, text_type=args.text_type, text=text, char_size=args.char_size, augment=args.augment,
        fonts_json=args.fonts_json, fonts_root=args.fonts_root,
        bad_font_file=args.bad_font_file, experiment_dir=args.experiment_dir,
        type_fonts=args.type_fonts, embedding_num=args.embedding_num, resume=args.resume, init_num=args.init_num,
        special_type=args.special_type, segment_type=args.segment_type
    )
    handle.generate_book_page_with_text()
