

export CUDA_VISIBLE_DEVICES=1

python data_generator/generate_specific_book_pages.py \
--obj_num 5 \
--text_type vertical \
--text_file raw_text/ci_qu_shi.txt \
--char_size 64 \
--augment \
--fonts_json /disks/sdb/projs/AncientBooks/data/chars/font_missing1.json \
--experiment_dir fz2_experiment \
--type_fonts type/方正第二批.txt \
--embedding_num 520 \
--resume 180000 \
--init_num 0 \
--special_type split \
