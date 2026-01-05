# # 英文空间过程执行任务
# python ./layer_sweep_probe_procedure.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"

# python ./layer_sweep_probe_procedure_fixed.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"

# # 中文空间过程执行任务
# python ./layer_sweep_probe_procedure_CN.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_CN_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_CN_test_with_prompt.json"

# python ./layer_sweep_probe_procedure_fixed_CN.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_CN_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_CN_test_with_prompt.json"

# # 阿拉伯语空间过程执行任务
# python ./layer_sweep_probe_procedure_AR.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json"

# python ./layer_sweep_probe_procedure_fixed_AR.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_with_prompt.json" \
#     -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json"


# 英文空间过程执行任务
python ./layer_sweep_probe_procedure_fixed.py \
    -m "meta-llama/Meta-Llama-3-8B-Instruct" \
    -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
    -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"

# 中文空间过程执行任务
python ./layer_sweep_probe_procedure_fixed.py \
    -m "meta-llama/Meta-Llama-3-8B-Instruct" \
    -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_with_prompt.json" \
    -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json"

# 阿拉伯语空间过程执行任务
python ./layer_sweep_probe_procedure_fixed.py \
    -m "meta-llama/Meta-Llama-3-8B-Instruct" \
    -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_with_prompt.json" \
    -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json"