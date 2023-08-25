
# for denoise

python ./uformer_restoration.py \
     --operation denoise \
     --model_weight ./model/denoise/Uformer_B.pth \
     --input_dir ./image_in/denoise_img/30_10.png \
     --output_dir ./image_out/denoise/ \
     --ps 128


# for deblur
<<com
python ./uformer_restoration.py \
    --operation deblur \
    --model_weight ./model/deblur/Uformer_B.pth \
    --input_dir ./image_in/deblur_img/GOPR0384_11_00-000018.png \
    --output_dir ./image_out/deblur \
    --ps 128
com

read -p "Press Enter to exit..."