python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/collage_mix_train.yaml \
--scale_lr False \
--name collage_mix_magic_fixup
