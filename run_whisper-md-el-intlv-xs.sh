# whisper-md-el-intlv-xs
python run_speech_recognition_seq2seq_streaming.py \
                --model_name_or_path   'openai/whisper-medium' \
                --model_revision   main \
                --do_train   True \
                --do_eval   True \
                --use_auth_token   False \
                --freeze_feature_encoder   False \
                --freeze_encoder   False \
                --model_index_name   'whisper-md-el-intlv-xs' \
                --dataset_name 'mozilla-foundation/common_voice_11_0,google/fleurs' \
                --dataset_config_name 'el,el_gr' \
                --train_split_name  'train+validation,train+validation' \
                --eval_split_name   'test,-' \
                --text_column_name  'sentence,transcription' \
                --audio_column_name 'audio,audio' \
                --streaming   False \
                --max_duration_in_seconds   30 \
                --do_lower_case   False \
                --do_remove_punctuation   False \
                --do_normalize_eval   True \
                --language   greek \
                --task transcribe \
                --shuffle_buffer_size   500 \
                --output_dir   './data/finetuningRuns/whisper-md-el-intlv-xs' \
                --overwrite_output_dir   True \
                --per_device_train_batch_size   32 \
                --gradient_accumulation_steps  2 \
                --learning_rate   1e-5 \
                --dropout         0.1 \
                --warmup_steps   500 \
                --max_steps   5000 \
                --eval_steps   1000 \
                --gradient_checkpointing   True \
                --cache_dir   '~/.cache' \
                --fp16   True \
                --evaluation_strategy   steps \
                --per_device_eval_batch_size   8 \
                --predict_with_generate   True \
                --generation_max_length   225 \
                --save_steps   1000 \
                --logging_steps   25 \
                --report_to   tensorboard \
                --load_best_model_at_end   True \
                --metric_for_best_model   wer \
                --greater_is_better   False \
                --push_to_hub   False 


#

