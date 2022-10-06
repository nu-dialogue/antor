# Reinforcement learning of NLG with PPO

## Setup WandB
We use [WandB](https://wandb.ai/site), which is a tool for visualizing and tracking ML experiments.
You must sing in and log in to wandb (see [WandB - Quickstart](https://wandb.ai/quickstart/python-script)).

## Fine-tune NLG with PPO
Run train.py with the arguments of your choice.
- Important Arguments
    - project_id: project id in WandB
    - run_id: run id in WandB; also used as checkpont name for weights fine-tuned via ppo
    - nlu_name: architecture type of NLU to be used
    - nlu_model_name: pre-trained model of NLU to be used
    - apply_noise: whether to apply ASR error simulation; if False, noise_type (see below) is ignored
    - noise_type: noise type to be used
- Example
    ```bash
    python ppo_train.py \
            --project_id 'milu-full-sys-noise' \
            --run_id 'milu-full-sys-noise-background(0)-seed12' \
            --nlu_name 'milu' \
            --nlu_model_name 'full-sys' \
            --apply_noise 'True' \
            --noise_type 'background(0)' \
            --random_seed '12' \
    ```
The checkponts will be saved to `outputs/<run_id>`. For the learning process, see wandb.