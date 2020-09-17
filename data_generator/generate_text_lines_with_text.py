
import os
import sys
import json
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from queue import Queue

from config import ONE_TEXT_LINE_IMGS_H, ONE_TEXT_LINE_TAGS_FILE_H
from config import ONE_TEXT_LINE_IMGS_V, ONE_TEXT_LINE_TAGS_FILE_V
from config import ONE_TEXT_LINE_TFRECORDS_H, ONE_TEXT_LINE_TFRECORDS_V
from config import TWO_TEXT_LINE_IMGS_H, TWO_TEXT_LINE_TAGS_FILE_H
from config import TWO_TEXT_LINE_IMGS_V, TWO_TEXT_LINE_TAGS_FILE_V
from config import TWO_TEXT_LINE_TFRECORDS_H, TWO_TEXT_LINE_TFRECORDS_V
from config import MIX_TEXT_LINE_IMGS_H, MIX_TEXT_LINE_TAGS_FILE_H
from config import MIX_TEXT_LINE_IMGS_V, MIX_TEXT_LINE_TAGS_FILE_V
from config import MIX_TEXT_LINE_TFRECORDS_H, MIX_TEXT_LINE_TFRECORDS_V
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
from data_generator.img_utils import generate_bigger_image_by_font, generate_bigger_image_by_shufa
from data_generator.img_utils import load_external_image_bigger
from data_generator.generate_chinese_images import get_external_image_paths


def check_text_type(text_type):
    if text_type.lower() in ("h", "horizontal"):
        text_type = "h"
    elif text_type.lower() in ("v", "vertical"):
        text_type = "v"
    else:
        ValueError("Optional text_types: 'h', 'horizontal', 'v', 'vertical'.")
    return text_type


class generate_text_lines_with_text_handle:
    def __init__(self, obj_num, font_path, shape=None, text_type="horizontal", text='野火烧不尽春风吹又生', char_size=64):
        self.text = Queue()
        for char in text:
            self.text.put(char)
        self.text_type = text_type
        if shape is None:
            self.shape = BOOK_PAGE_SHAPE_LIST
        elif isinstance(shape, tuple) and len(shape) == 2:
            self.shape = shape
        else:
            raise ValueError('shape should be tuple and length 2')
        self.obj_num = obj_num
        self.font_path = []
        for file in os.listdir(font_path):
            ext = os.path.splitext(file)[-1].lower()
            if ext in ['ttf', 'otf']:
                self.font_path.append(os.path.join(font_path, file))
        if not len(self.font_path):
            raise ValueError('Find 0 font file in {}'.format(font_path))
        self.char_size = char_size

    def generate_book_page_with_text(self, init_num=0):
        text_type = check_text_type(self.text_type)

        if text_type == "h":
            book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
        elif text_type == "v":
            book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
        else:
            raise ValueError('text_type should be horizontal or vertical')

        check_or_makedirs(book_page_imgs_dir)
        obj_num = self.obj_num

        with open(book_page_tags_file, "w", encoding="utf-8") as fw:
            for i in range(init_num, init_num + obj_num):
                if isinstance(self.shape, list):
                    shape = random.choice(self.shape)
                else:
                    shape = self.shape

                font = random.choice(self.font_path)
                font = ImageFont.truetype(font, size=self.char_size)

                PIL_page, text_bbox_list, text_list = self.create_book_page_with_text(
                    shape,
                    text_type=text_type,
                    font=font
                )

                image_tags = {"text_bbox_list": text_bbox_list, "text_list": text_list}

                img_name = "book_page_%d.jpg" % i
                save_path = os.path.join(book_page_imgs_dir, img_name)
                PIL_page.save(save_path, format="jpeg")
                fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")

                if i % 50 == 0:
                    print(" %d / %d Done" % (i, obj_num))
                    sys.stdout.flush()

    def create_book_page_with_text(self, shape, text_type, font):
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
                text_bbox_list, text_list, text = self.generate_mix_rows_chars_with_text(
                    x, y1, y2, row_length, np_page, char_spacing
                )
                text_bbox_records_list.extend(text_bbox_list)
                text_records_list.extend(text_list)

        else:  # 纵向排列

            # 随机决定文本的列数
            # cols_num = random.randint(6, 10)
            # cols_num = random.randint(18, 23)
            # cols_num = random.randint(14, 19)
            cols_num = random.randint(16, 20)
            col_w = (page_width - 2 * margin_w) / cols_num

            # x-coordinate划分列
            xs = [margin_w + round(i * col_w) for i in range(cols_num)] + [page_width - 1 - margin_w, ]

            # 画列线，第一条线和最后一条线是边缘线，不需要画
            if draw is not None:
                for x in xs[1:-1]:
                    draw.line([(x, margin_h), (x, page_height - 1 - margin_h)], fill="white", width=line_thickness)
                np_page = np.array(PIL_page, dtype=np.uint8)

            # 随机决定字符间距占列距的比例
            char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)

            # 逐列生成汉字，最右边为第一列
            for i in range(len(xs) - 1, 0, -1):
                x1, x2 = xs[i - 1] + 1, xs[i] - 1
                y = margin_h + int(random.uniform(0.0, 1) * margin_line_thickness)
                col_length = page_height - y - margin_h
                text_bbox_list, text_list, text = self.generate_mix_cols_chars_with_text(
                    x1, x2, y, col_length, np_page, char_spacing
                )
                text_bbox_records_list.extend(text_bbox_list)
                text_records_list.extend(text_list)

        # 将黑底白字转换为白底黑字
        np_page = reverse_image_color(np_img=np_page)
        PIL_page = Image.fromarray(np_page)

        # print(text_bbox_list)
        # print(len(text_bbox_list))
        # PIL_page.show()

        return PIL_page, text_bbox_records_list, text_records_list

    def generate_mix_rows_chars_with_text(self, x, y1, y2, row_length, np_background, char_spacing):
        row_height = y2 - y1 + 1
        x_start = x

        text_bbox_list = []
        text_list = []
        flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
        remaining_len = row_length
        while remaining_len >= row_height:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(row_height, remaining_len)
            flag += 1
            if flag % 2 == 1:
                x, text_bbox, _, _ = generate_one_row_chars(x, y1, y2, length, np_background, char_spacing)
                text_bbox_list.append(text_bbox)
                head_tail_list.append((text_bbox[0], text_bbox[2]))
            else:
                x, text1_bbox, text2_bbox, _ = generate_two_rows_chars(x, y1, y2, length, np_background, char_spacing,
                                                                       use_img)
                text_bbox_list.extend([text1_bbox, text2_bbox])
                head_tail_list.append((min(text1_bbox[0], text2_bbox[0]), max(text1_bbox[2], text2_bbox[2])))
            remaining_len = row_length - (x - x_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        return x, text_bbox_list, split_pos

    def generate_mix_cols_chars_with_text(self, x1, x2, y, col_length, np_background, char_spacing):
        col_width = x2 - x1 + 1
        y_start = y

        text_bbox_list = []
        text_list = []
        flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
        remaining_len = col_length
        while remaining_len >= col_width:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(col_width, remaining_len)
            flag += 1
            if flag % 2 == 1:
                y, text_bbox, _, _ = generate_one_col_chars(x1, x2, y, length, np_background, char_spacing)
                text_bbox_list.append(text_bbox)
            else:
                y, text1_bbox, text2_bbox, _ = generate_two_cols_chars(x1, x2, y, length, np_background, char_spacing)
                text_bbox_list.extend([text1_bbox, text2_bbox])
            remaining_len = col_length - (y - y_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        # 获取单双行的划分位置
        char_spacing_h = round(col_width * char_spacing[0])
        head_y1, tail_y2 = head_tail_list[0][0], head_tail_list[-1][1]
        split_pos = [head_y1, ]
        for i in range(len(head_tail_list) - 1):
            y_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
            split_pos.append(y_cent)
        split_pos.append(tail_y2)

        return y, text_bbox_list, split_pos


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


def generate_char_img_into_unclosed_box_with_text(
        np_background,
        x1, y1, x2=None, y2=None,
        char_spacing=(0.05, 0.05),
        text=''):
    if x2 is None and y2 is None:
        raise ValueError("There is one and only one None in (x2, y2).")
    if x2 is not None and y2 is not None:
        raise ValueError("There is one and only one None in (x2, y2).")

    # 生成白底黑字的字，包含文字
    if text:
        chinese_char = text[0]
    else:
        chinese_char = ' '

    PIL_char_img = generate_single_char(chinese_char, font)

    # 随机决定是否对汉字图片进行旋转，以及旋转的角度
    if random.random() < 0.35:
        PIL_char_img = rotate_PIL_image(PIL_char_img, rotate_angle=random.randint(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE))

    # 转为numpy格式
    np_char_img = np.array(PIL_char_img, dtype=np.uint8)

    if chinese_char in IMPORTANT_CHARS:
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
        np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = np_char_img
    except ValueError as e:
        # print('Exception:', e)
        # print("The size of char_img is larger than the length of (y1, x1) to edge. Now, resize char_img ...")
        if x2 is None:
            box_x2 = np_background.shape[1] - 1
            box_w = box_x2 - box_x1 + 1
        else:
            box_y2 = np_background.shape[0] - 1
            box_h = box_y2 - box_y1 + 1
        np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = np_char_img

    # 包围汉字的最小box作为bounding-box
    # bounding_box = (box_x1, box_y1, box_x2, box_y2)

    # 随机选定汉字图片的bounding-box
    bbox_x1 = random.randint(x1, box_x1)
    bbox_y1 = random.randint(y1, box_y1)
    bbox_x2 = min(random.randint(box_x2, box_x2 + char_spacing_w), np_background.shape[1] - 1)
    bbox_y2 = min(random.randint(box_y2, box_y2 + char_spacing_h), np_background.shape[0] - 1)
    bounding_box = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    char_box_tail = box_x2 + 1 if x2 is None else box_y2 + 1

    return chinese_char, bounding_box, char_box_tail


def generate_one_row_chars_with_text(x, y1, y2, length, np_background, char_spacing, text=''):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    row_end = x + length - 1
    row_height = y2 - y1 + 1
    while length >= row_height:
        chinese_char, bounding_box, x_tail = generate_char_img_into_unclosed_box_with_text(
            np_background,
            x1=x, y1=y1,
            x2=None, y2=y2,
            char_spacing=char_spacing,
            text=text
        )

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = x_tail - x
        length -= added_length
        x = x_tail

    # 获取文本行的bounding-box
    head_x1, head_y1, _, _ = char_and_box_list[0][1]
    _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)

    # 获取字符之间的划分位置
    char_spacing_w = round(row_height * char_spacing[1])
    split_pos = [head_x1, ]
    for i in range(len(char_and_box_list) - 1):
        x_cent = (char_and_box_list[i][1][2] + char_and_box_list[i + 1][1][0]) // 2
        split_pos.append(x_cent)
    split_pos.append(tail_x2)

    return x, text_bbox, char_and_box_list, split_pos


def generate_two_rows_chars_with_text(x, y1, y2, length, np_background, char_spacing, text=''):
    row_height = y2 - y1 + 1
    mid_y = y1 + round(row_height / 2)

    x_1, text1_bbox, _, _ = generate_one_row_chars(x, y1, mid_y, length, np_background, char_spacing, use_img)
    x_2, text2_bbox, _, _ = generate_one_row_chars(x, mid_y + 1, y2, length, np_background, char_spacing, use_img)

    # 获取文本行之间的划分位置
    center_val = (text1_bbox[3] + text2_bbox[1]) // 2
    char_spacing_h = round(row_height * char_spacing[0])
    split_pos = [text1_bbox[1], center_val, text2_bbox[3]]

    return max(x_1, x_2), text1_bbox, text2_bbox, split_pos


def generate_one_col_chars(x1, x2, y, length, np_background, char_spacing, text=''):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    col_end = y + length - 1
    col_width = x2 - x1 + 1
    while length >= col_width:
        chinese_char, bounding_box, y_tail = generate_char_img_into_unclosed_box(
            np_background,
            x1=x1, y1=y,
            x2=x2, y2=None,
            char_spacing=char_spacing,
            use_img=use_img
        )

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = y_tail - y
        length -= added_length
        y = y_tail

    # 获取文本行的bounding-box
    head_x1, head_y1, _, _ = char_and_box_list[0][1]
    _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)

    # 获取字符之间的划分位置
    char_spacing_h = round(col_width * char_spacing[0])
    split_pos = [head_y1, ]
    for i in range(len(char_and_box_list) - 1):
        x_cent = (char_and_box_list[i][1][3] + char_and_box_list[i + 1][1][1]) // 2
        split_pos.append(x_cent)
    split_pos.append(tail_y2)

    return y, text_bbox, char_and_box_list, split_pos


def generate_two_cols_chars(x1, x2, y, length, np_background, char_spacing, text=''):
    col_width = x2 - x1 + 1
    mid_x = x1 + round(col_width / 2)

    y_1, text1_bbox, _, _ = generate_one_col_chars(x1, mid_x, y, length, np_background, char_spacing, use_img)
    y_2, text2_bbox, _, _ = generate_one_col_chars(mid_x + 1, x2, y, length, np_background, char_spacing, use_img)

    # 获取文本行之间的划分位置
    center_val = (text1_bbox[2] + text2_bbox[0]) // 2
    char_spacing_w = round(col_width * char_spacing[1])
    split_pos = [text1_bbox[0], center_val, text2_bbox[2]]

    return max(y_1, y_2), text1_bbox, text2_bbox, split_pos

