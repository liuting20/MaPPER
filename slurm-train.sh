#!/bin/bash


MASTER_PORT=29560 \
OMP_NUM_THREADS=$omp \

torchrun --nproc_per_node=8 --master_port=29999 ./train.py \
--batch_size 32 \
--lr_bert 0.00001 \
--aug_crop \
--aug_scale \
--aug_translate \
--backbone resnet50 \
--bert_enc_num 12 \
--detr_enc_num 6 \
--dataset coco+ \
--max_query_len 20 \
--output_dir outputs-mapper/coco+ \
--foundation_model_path ./checkpoints/dinov2_vitb14_reg4_pretrain.pth \
--epoch 180 \
--lr_drop 120 \

