#!/bin/bash
python /home/qdx/1-MACE/mace/6-AL_flow/activate_learning_full_flow.py \
    --data_path data \
    --foundation_model /home/qdx/1-MACE/mace/6-AL_flow/mace-mpa-0-medium.model \
    --subset_size 1000 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42 \
    --n_comm 4 \
    --device cuda \
    --comm_epoch 100\
    --md_temp 300 \
    --md_steps 10 \
    --uncert_threshold 0.9 \
    --md_write_interval 10 \
    --uncert_threshold 0.9 \
    --uncert_max_frames 100 \
    --n_rounds 1 \
    --dft_root dft_cal \
    --dft_template_dir dft_input \
    --submitjobs submitjobs.sh \
    --outcar2xyz outcar2xyz.sh \
    --dft_poll_sec 300 \
    --dft_timeout_hr 24.0 \
    --merged_dft_name dft_merged.extxyz \





    