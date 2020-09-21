
import os
import sys

import json

if __name__ == '__main__':
    img_name_list = []
    bbox_list = []
    with open('data/book_pages/book_pages_tags_vertical_3.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            res = line.strip().split('\t')
            img_name_list.append(res[0])
            bbox_list.append(json.loads(res[1]))
    gt_file = 'data/book_pages/gt'
    os.makedirs(gt_file, exist_ok=True)
    for img_name, bbox in zip(img_name_list, bbox_list):
        with open(os.path.join(gt_file, 'gt_' + os.path.splitext(img_name)[0] + '.txt'), 'w', encoding='utf-8') as fp:
            for text_bbox, text in zip(bbox['text_bbox_list'], bbox['text_list']):
                text_bbox = list(map(str, text_bbox))
                fp.write(' '.join(text_bbox))
                fp.write(' ')
                fp.write('\"' + ''.join(text) + '\"')
                fp.write('\n')
