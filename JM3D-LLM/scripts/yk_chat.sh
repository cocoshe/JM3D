# MODEL_PATH='/home/myw/wuchangli/PointLLM/outputs/PointLLM_train_stage1/colorv1.1_v5'
MODEL_PATH='/home/myw/wuchangli/yk/JM3D/JM3D-LLM/checkpoints/yk_test_ckpt_backup'
MODEL_PATH='/home/myw/wuchangli/yk/JM3D/JM3D-LLM/checkpoints/yk_test_ckpt_stage2'


CUDA_VISIBLE_DEVICES=2 \
    python llava/eval/run_llava_yk.py \
    --model_path $MODEL_PATH \
    --data_path data/objaverse_data \
    --torch_dtype float32