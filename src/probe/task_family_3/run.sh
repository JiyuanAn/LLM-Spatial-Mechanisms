python ./layer_sweep_probe_procedure.py \                                                                                                                                                                                       main ✗
    -m "Qwen/Qwen2.5-7B-Instruct" \
    -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
    -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"

python ./layer_sweep_probe_procedure_fixed.py \                                                                                                                                                                                 main ✗
    -m "Qwen/Qwen2.5-7B-Instruct" \
    -tr "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
    -te "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_3/data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"