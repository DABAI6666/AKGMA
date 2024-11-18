#!/bin/bash

CKPT="AKGMA-LLaVA"
STEP="1epoch"
SPLIT="mscoco_val"

# Run the model caption loader with the updated arguments for MSCOCO
python -m llava.eval.model_caption_loader \
      --model-base lmsys/vicuna-13b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/image_caption/eval/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/image_caption/eval/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode plain &

wait

# Evaluate the predictions using the updated MSCOCO evaluation script
python playground/image_caption/eval/mscoco/mscoco_eval.py --pred_file ./playground/image_caption/eval/answers/$SPLIT/$CKPT/$STEP.jsonl