dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_focal_gamma01
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


# "0.2 0.4 0.6 0.8 1.0"

for gamma in "1.0" "2.0" "4.0"
do
    echo "class focal : $gamma"

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
    --exp_id ${exp_id} \
    --cls_both \
    --m_classes "[10, 30, 70, 150]" \
    --label_loss_type "focal" \
    --focal_gamma ${gamma} \
    --aux_label_loss_type "ce" \
    ${@:1}

done

for gamma in "1.0" "2.0" "4.0"
do
    echo "class focal : $gamma"

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
    --exp_id ${exp_id} \
    --cls_both \
    --m_classes "[10, 30, 70, 150]" \
    --label_loss_type "focal" \
    --focal_gamma ${gamma} \
    --aux_label_loss_type "ce" \
    --set_cost_class 2\
    ${@:1}

done