date
python ./run_speech_recognition_seq2seq_streaming.py \
                --model_name_or_path   "openai/whisper-small" \
                --model_revision       "main" \
                --do_train             True \
                --do_eval              True \
                --use_auth_token       False \
                --freeze_encoder       False \
                --model_index_name     "whisper-sm-el-xs" \
                --dataset_name         "mozilla-foundation/common_voice_11_0" "google/fleurs" \
                --dataset_config_name  "el" \
                --audio_column_name    "audio" \
                --text_column_name     "sentence" \
                --max_duration_in_seconds 30 \
                --train_split_name    "train+validation" \
                --eval_split_name      "test" \
                --do_lower_case         False \
                --do_remove_punctuation False \
                --do_normalize_eval     True \
                --language              "greek" \
                --task                  "transcribe" \
                --shuffle_buffer_size   500 \
                --output_dir             "./data/finetuningRuns/whisper-sm-el-xs-3k" \
                --per_device_train_batch_size 16 \
                --gradient_accumulation_steps 4  \
                --learning_rate          1e-5 \
                --dropout                0.1 \
                --warmup_steps           500 \
                --max_steps              3000 \
                --gradient_checkpointing True \
                --fp16                   True \
                --evaluation_strategy    "steps" \
                --per_device_eval_batch_size 8 \
                --predict_with_generate  True \
                --generation_max_length  225 \
                --save_steps             1000 \
                --eval_steps             1000 \
                --logging_steps          25 \
                --report_to              "tensorboard" \
                --load_best_model_at_end True \
                --metric_for_best_model  "wer" \
                --greater_is_better      False \
                --push_to_hub            False \
                --overwrite_output_dir    True 


date