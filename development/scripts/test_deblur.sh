
python ./test/test_deblur.py \
    --test_dir ./datasets/deblur/GoPro/customized_dataset/test \
    --result_dir ./result_dir/test/deblur \
    --pretrain_weights ./models/training/deblur/model_best.pth \
    --do_validation \
    --test_ps 128 \
    --gpu 0 \
    --batch_size 1


read -p "Press Enter to exit..."