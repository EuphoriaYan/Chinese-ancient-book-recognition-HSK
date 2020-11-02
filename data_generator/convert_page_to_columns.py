import argparse
import os
import json
from PIL import Image
from random import random


def convert_page_to_columns(input_dir, train_output_dir, val_output_dir):
    input_gt = os.path.join(input_dir, 'book_pages_tags_vertical_3.txt')
    pic_gt = []
    with open(input_gt, 'r', encoding='utf-8') as gt:
        for line in gt:
            img_name, gt_json = line.strip().split('\t')
            pic_gt.append((img_name, json.loads(gt_json)))
    os.makedirs(os.path.join(train_output_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(val_output_dir, 'imgs'), exist_ok=True)
    with open(os.path.join(train_output_dir, 'gt_file.txt'), 'w', encoding='utf-8') as train_gt_file, \
            open(os.path.join(val_output_dir, 'gt_file.txt'), 'w', encoding='utf-8') as val_gt_file:
        for img_name, img_json in pic_gt:
            img = Image.open(os.path.join(input_dir, 'imgs_vertical', img_name))
            bbox_list = img_json['text_bbox_list']
            text_list = img_json['text_list']
            base_name, ext = os.path.splitext(img_name)
            assert len(bbox_list) == len(text_list)
            for i, (bbox, text) in enumerate(zip(bbox_list, text_list)):
                crop_img = img.crop(bbox)
                save_dir = os.path.join('imgs', base_name + '_' + str(i) + ext)
                if random() < 0.995:
                    crop_img.save(os.path.join(train_output_dir, save_dir))
                    train_gt_file.write(save_dir + '\t' + ''.join(text) + '\n')
                else:
                    crop_img.save(os.path.join(val_output_dir, save_dir))
                    val_gt_file.write(save_dir + '\t' + ''.join(text) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--train_output_dir', type=str, required=True)
    parser.add_argument('--val_output_dir', type=str, required=True)
    args = parser.parse_args()
    convert_page_to_columns(args.input_dir, args.train_output_dir, args.val_output_dir)