# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
export WANDB_PROJECT= t5_base_nl2api
export BS=64
export GAS=1
export m=t5-base
export MAX_LEN=128
export DATA_DIR=T5_data/email/horizontal
export OUTPUT_DIR = api_model/email_horizontal_config_0
export MAX_TGT_LEN=128
python finetune_trainer.py \
    --tokenizer_name $m --model_name_or_path $m \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR --overwrite_output_dir \
    --learning_rate=3e-4 --adafactor\
    --warmup_steps 500 --sortish_sampler \
    --gradient_accumulation_steps=$GAS \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=6 \
    --save_steps 3000 --eval_steps 3000 \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN \
    --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --logging_first_step \
    --task translation --label_smoothing_factor 0.1 \
    "$@"
