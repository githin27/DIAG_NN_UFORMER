
python ./test/test_denoise.py \
    --test_dir ./datasets/denoise/SIDD/customized_dataset/test \
    --result_dir ./result_dir/test/denoise \
    --pretrain_weights ./models/training/denoise/model_best.pth \
    --test_ps 128 \
    --gpu 0 \
    --batch_size 1 



read -p "Press Enter to exit..."