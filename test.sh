export CUDA_VISIBLE_DEVICES=0


# # RefCOCO
python eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --dataset unc --max_query_len 20 --eval_set testA --eval_model ./outputs-mapper/coco/best_checkpoint.pth --output_dir ./outputs-test/refcoco_testa


# # RefCOCO+
# python eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testB --eval_model ./outputs-mapper/coco+/best_checkpoint.pth --output_dir ./outputs-test/refcoco_plus_r50_b

# # RefCOCOg g-split
# python eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ./outputs-mapper/cocog-u/best_checkpoint.pth --output_dir ./outputs-test/refcocog_gsplit_r50

# # RefCOCOg u-split
# python eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs-mapper/cocog-u/best_checkpoint.pth --output_dir ./outputs-test/refcocog_usplit_r50
