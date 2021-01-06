export CUDA_VISIBLE_DEVICES=0

python data_generator/generate_specific_book_pages.py \
--obj_num 100000 \
--text_type vertical \
--text_file raw_text/charset_cover.txt \
--char_size 64 \
--augment \
--fonts_json /data/pku-orc-train/fonts/font_missing1.json \
--fonts_root /data/pku-orc-train/fonts/FangZ2 \
--experiment_dir ./fz2_experiment \
--type_fonts type/方正第二批.txt \
--embedding_num 520 \
--resume 180000 \
--init_num 0 \
--special_type normal \
--segment_type mixed \

# segment_type: normal/spacious/crowded/mixed
