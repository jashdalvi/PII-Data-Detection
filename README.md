## Solution for Kaggle Competition (41st Position Silver medal) - [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

## Steps to run the code

### Setup the env

```
pip install -r requirements.txt
```

### Change the directory

```
cd src
```

### Run the code

For training on a single GPU with mixed precision

```python
accelerate launch --mixed_precision=fp16 train_accelerate.py fold=0
```

For training on a Multi-GPU with mixed precision
<br>Note: Make sure to change the num of processes according to the number of GPUs you have. The following command is for 2 GPUs.

```python
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_accelerate.py fold=0
```

Take a look at the src/config/config_accelerate.py file for changing configuration details.
Also add your wandb API key as an ENV variable to log the metrics on wandb.

For training Llama-3-8b LLM on multiple GPUs. The configuration uses LoRA during finetuning.

```python
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 train_llm.py fold=2 upload_models=True
```
