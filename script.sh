export CUDA_VISIBLE_DEVICES=0
# Train pose estimator
python -m engine.train_estimator --model_save your_save_path --num_workers 20 --batch_size 16 --train_steps 1500 --seed 1677330429 \
    --dataset_dir Your_data_path \
    --total_epoch 300 \
    --use_clip 1.0 --feat_c_R 1286 --feat_c_ts 1289 --use_clip_global 1.0 --use_clip_atte 1.0 --heads 2 \
    --dataset your_data_type    

# Evaluate the trained model
python -m evaluation.evaluate  --model_save your_save_path \
    --resume 1 --resume_model your_save_path --dataset 'Real' \
    --detection_dir your_data/segmentation_results \
    --dataset_dir your_data_pth \
    --use_clip 1.0 --feat_c_R 1286 --feat_c_ts 1289 --use_clip_global 1.0 --use_clip_atte 1.0 --heads 2 \


