export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
# ReferItGame
# python train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50


# # RefCOCO  学习率加大了10倍
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130

# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 130

# torchrun --nproc_per_node=8 train.py 
# torchrun --nproc_per_node=8 train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 200 --lr_drop 120

# torchrun --nproc_per_node=8 train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs-mapper/coco+ --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 200 --lr_drop 120

# python train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130 

# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120
# python train.py --batch_size 2 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco+_r50 --epochs 180 --lr_drop 120 --foundation_model_path /share/home/liuting/place_recog/SelaVPR-main/pretrain_model/dinov2_vitb14_reg4_pretrain.pth


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130

# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50

python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --bert_enc_num 12 --dataset gref --max_query_len 40 --output_dir outputs-mapper/refcocog_gsplit_r50 --foundation_model_path /share/home/liuting/transvg_data/checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 130


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50
