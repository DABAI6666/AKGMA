#!/bin/bash

nohup deepspeed decoder_model/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --lm_model_path /home/Cognitive-Visual-Language-Mapper-main/opt-125m \
    --clip_model_path /home/Cognitive-Visual-Language-Mapper-main/clip-vit-large-patch14-336 \
    --data_path playground/knowledge_data/Wikipedia_2M.json \
    --image_folder playground/knowledge_data/wikipedia_images_2m \
    --triple_data_path playground/knowledge_data/wikipedia_images_2m_triple.json \
    --output_dir ./checkpoints/opt-pretrain \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 12 > pretrain_knowledge.log 2>&1 & \