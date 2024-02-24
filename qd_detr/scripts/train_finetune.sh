dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_finetune
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

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

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id "tgt_cc" \
# --m_classes "[10, 30, 70, 150]" \
# --tgt_embed \
# --cc_matching \
# --resume "results_pretrain_0/hl-video_tef-pt-2024_02_23_01_13_27/model_latest.ckpt"
# ${@:1}

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id "org" \
# --resume "results_pretrain_1/hl-video_tef-pt-2024_02_22_18_37_15/model_latest.ckpt" \
# ${@:1}

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
--exp_id "org_40" \
--num_queries 40
--resume "results_pretrain_1/hl-video_tef-pt-2024_02_23_03_34_44/model_latest.ckpt" \
${@:1}




# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --a_feat_dir ${a_feat_dir} \
# --a_feat_dim ${a_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id org \

# ${@:1}



# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --a_feat_dir ${a_feat_dir} \
# --a_feat_dim ${a_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id ${exp_id} \
# --m_classes "[10, 30, 70, 150]" \
# --cls_both \
# --tgt_embed \
# --class_anchor \
# --cc_matching \
# --set_cost_class 2 \
# --label_loss_coef 2 \
# ${@:1}

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --a_feat_dir ${a_feat_dir} \
# --a_feat_dim ${a_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id ${exp_id} \
# --m_classes "[10, 30, 70, 150]" \
# --cls_both \
# --tgt_embed \
# --class_anchor \
# --cc_matching \
# --set_cost_class 2 \
# --label_loss_coef 2 \
# --label_loss_type "focal" \
# --focal_gamma "1.0" \
# --aux_label_loss_type "focal" \
# --aux_focal_gamma "1.0" \
# ${@:1}

