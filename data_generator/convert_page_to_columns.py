import argparse
import os
import json
from PIL import Image


def convert_page_to_columns(input_dir, output_dir):
    input_gt = os.path.join(input_dir, 'book_pages_tags_vertical_3.txt')
    pic_gt = []
    with open(input_gt, 'r', encoding='utf-8') as gt:
        for line in gt:
            img_name, gt_json = line.strip().split('\t')
            pic_gt.append((img_name, json.loads(gt_json)))
    os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
    with open(os.path.join(output_dir, 'gt_file.txt'), 'w', encoding='utf-8') as gt_file:
        for img_name, img_json in pic_gt:
            img = Image.open(os.path.join(input_dir, 'imgs_vertical', img_name))
            bbox_list = img_json['text_bbox_list']
            text_list = img_json['text_list']
            base_name, ext = os.path.splitext(img_name)
            assert len(bbox_list) == len(text_list)
            for i, (bbox, text) in enumerate(zip(bbox_list, text_list)):
                crop_img = img.crop(bbox)
                save_dir = os.path.join('imgs', base_name + '_' + str(i) + ext)
                crop_img.save(os.path.join(output_dir, save_dir))
                gt_file.write(save_dir)
                gt_file.write('\t')
                gt_file.write(''.join(text))
                gt_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    convert_page_to_columns(args.input_dir, args.output_dir)
