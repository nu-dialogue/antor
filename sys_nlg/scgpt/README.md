# SCGPT
- Paper: [Few-shot Natural Language Generation for Task-Oriented Dialog](https://arxiv.org/abs/2002.12328)
- GitHub: [pengbaolin/SC-GPT](https://github.com/pengbaolin/SC-GPT)

## 1. Preprocess MultiWOZ data
We use the data format for scgpt model used on ConvLab2. Execute the `preprocess.py`, which is a modification of `ConvLab-2/convlab2/nlg/scgpt/multiwoz/preprocess.py` for system response generation instead of user utterance generation.
```bash
python preprocess.py
```

## 2. Prepare pre-trained model
We use the official pre-trainined model for our experiment. Save the checkpoint in this directory with the following command.
```bash
wget https://bapengstorage.blob.core.windows.net/fileshare/scgpt.tar.gz
tar -xvf scgpt.tar.gz
```

## 3. Fine-tune SCGPT on MultiWOZ
For training script, we use `train.py` from the offical SCGPT repo.
```bash
CUDA_VISIBLE_DEVICES=0
python train.py --output_dir=multiwoz_finetuned \
                --model_type=gpt2 \
                --model_name_or_path=scgpt \
                --do_train \
                --train_data_file=data/train.txt \
                --per_gpu_train_batch_size 8 \
                --do_eval \
                --eval_data_file=data/val.txt \
                --per_gpu_eval_batch_size 16 \
                --num_train_epochs 5 \
                --learning_rate 5e-5 \
                --overwrite_cache \
                --use_tokenize \
                --overwrite_output_dir \
```