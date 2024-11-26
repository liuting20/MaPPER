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
--foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth \
--epoch 200 \
--lr_drop 120 \



# coco+
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=30000 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs-mapper/coco+ --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 200 --lr_drop 120

# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/coco+ --foundation_model_path /share/home/liuting/place_recog/SelaVPR-main/pretrain_model/dinov2_vitb14_reg4_pretrain.pth --epoch 200 --resume ./outputs/coco+/best_checkpoint.pth --lr_drop 120


# coco
# torchrun --nproc_per_node=8 train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/coco --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130

# coco
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --dataset unc --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130

# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/cocog-g-6.14 --foundation_model_path /share/home/liuting/place_recog/SelaVPR-main/pretrain_model/dinov2_vitb14_reg4_pretrain.pth --epochs 130 --resume ./outputs/cocog-g/best_checkpoint.pth 
