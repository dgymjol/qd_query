dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_charadesSTA2
exp_id=exp

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features/charades

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32
eval_bsz=32
lr_drop=400

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id "org_30" \
--max_v_l -1 \
--clip_length 1 \
--lr 0.0002 \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--eval_bsz ${eval_bsz} \
--num_queries 30 \
${@:1}


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id "_tgt_cc_10_20_150" \
--max_v_l -1 \
--clip_length 1 \
--lr 0.0002 \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--m_classes "[10, 20, 150]" \
--tgt_embed \
--cc_matching \
--eval_bsz ${eval_bsz} \
${@:1}


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id "_tgt_cc_20_150" \
--max_v_l -1 \
--clip_length 1 \
--lr 0.0002 \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--m_classes "[20, 150]" \
--tgt_embed \
--cc_matching \
--eval_bsz ${eval_bsz} \
${@:1}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id "org_20" \
--max_v_l -1 \
--clip_length 1 \
--lr 0.0002 \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--eval_bsz ${eval_bsz} \
--num_queries 20 \
${@:1}