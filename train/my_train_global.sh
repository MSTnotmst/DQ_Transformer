python my_train.py \
--name painter_global \
--gpu_ids 0 \
--model painter_global \
--dataset_mode null \
--batch_size 4 \
--global_res 256 \
--n_queries 400 \
--gt_strokes 64 \
--old_strokes 24 \
--extra_down 2 \
--num_blocks 3 \
--display_id 0 \
--display_freq 25 \
--print_freq 8 \
--lr 1e-4 \
--init_type normal \
--n_epochs 420 \
--n_epochs_decay 120 \
--max_dataset_size 256 \
--save_epoch_freq 20
# v1：純全域 straight 先跑通、確認去格。之後要疊 3A/3C/1C 再加 --real_brush / --curved_stroke。
# 顯存吃緊就降 --batch_size / --n_queries / --global_res，或把 --extra_down 設 3（token 數再 /4）。
