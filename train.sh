export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

# # RefCOCO  
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num --dataset unc --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path ./checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 100

# # RefCOCO  
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num --dataset unc+ --max_query_len 20 --output_dir outputs-mapper/coco --foundation_model_path ./checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 180 --lr_drop 120


# # RefCOCOg g-split
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --bert_enc_num 12 --dataset gref --max_query_len 40 --output_dir outputs-mapper/refcocog_gsplit_r50 --foundation_model_path ./checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 100


# # RefCOCOg umd-split
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --bert_enc_num 12 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50 --foundation_model_path ./checkpoints/dinov2_vitb14_reg4_pretrain.pth --epoch 100
