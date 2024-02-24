accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_accelerate.py fold=0 upload_models=False
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_accelerate.py fold=1 upload_models=False
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_accelerate.py fold=2 upload_models=False
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_accelerate.py fold=3 upload_models=True