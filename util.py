# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import shutil
import pandas as pd
from chinese_components.char2compo import chinese_components_info

from config import CHINESE_LABEL_FILE_S, CHINESE_LABEL_FILE_ML, TRADITION_CHARS_FILE
from config import IGNORABLE_CHARS_FILE, IMPORTANT_CHARS_FILE


def check_or_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def remove_then_makedirs(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


# json字典的key只能是字符串，python字典的key可以是 str, int, float, tuple
def chinese_labels_dict(charset_size='m'):
    assert os.path.exists(CHINESE_LABEL_FILE_S), "Charset file does not exist!"
    assert os.path.exists(CHINESE_LABEL_FILE_ML), "Charset file does not exits!"

    if charset_size == 's':
        with open(CHINESE_LABEL_FILE_S, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        lines = [line.strip() for line in lines]
        id2char_dict = {i: k for i, k in enumerate(lines)}
        char2id_dict = {k: i for i, k in enumerate(lines)}
        num_chars = len(char2id_dict)
    elif charset_size == 'm':
        charset_csv = pd.read_csv(CHINESE_LABEL_FILE_ML)
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
        id2char_dict = {i: k for i, k in enumerate(charset)}
        char2id_dict = {k: i for i, k in enumerate(charset)}
        num_chars = len(char2id_dict)
    elif charset_size == 'l':
        charset_csv = pd.read_csv(CHINESE_LABEL_FILE_ML)
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.9999].values.squeeze(axis=-1).tolist()
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
        id2char_dict = {i: k for i, k in enumerate(charset)}
        char2id_dict = {k: i for i, k in enumerate(charset)}
        num_chars = len(char2id_dict)
    else:
        raise ValueError('charset_size should be s, m or l.')

    return id2char_dict, char2id_dict, num_chars


def ignorable_chars():
    chars = set()
    with open(IGNORABLE_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return "".join(chars)


def important_chars():
    chars = set()
    with open(IMPORTANT_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return "".join(chars)


# General tasks
ID2CHAR_DICT, CHAR2ID_DICT, NUM_CHARS = chinese_labels_dict()
BLANK_CHAR = ID2CHAR_DICT[0]
IGNORABLE_CHARS = ignorable_chars()
IMPORTANT_CHARS = important_chars()


def traditional_chars():
    with open(TRADITION_CHARS_FILE, "r", encoding="utf-8") as fr:
        tradition_chars = fr.read()
        tradition_chars = tradition_chars.strip()
    tradition_chars = "".join([c for c in tradition_chars if c in CHAR2ID_DICT])
    return tradition_chars


TRADITION_CHARS = traditional_chars()

# Chinese components recognition task
CHAR_TO_COMPO_SEQ, COMPO_SEQ_TO_CHAR, NUM_CHAR_STRUC, NUM_SIMPLE_CHAR, NUM_LR_COMPO = chinese_components_info()

if __name__ == '__main__':
    # print(ignorable_chars())
    print("Done !")
