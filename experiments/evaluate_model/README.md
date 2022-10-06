# Evaluate NLG model
Test fine-tuned NLG with PPO and baseline methods

## Evaluate fine-tuned NLG (GPT-2)
Run `evaluate_ppo.py` with the **run_id** used in `experiments/ppo`. The last checkpoint in `experiments/ppo/outputs/<run_id>` is used as the model weight.
- Examples
    ```bash
    python evaluate_ppo.py --run_id 'milu-full-sys-noise-background(0)-seed12'
    ```
The results will be output to `outputs/ppo/<run_id>`.

## Evaluate baselines
Run `evaluate_baselines.py` with the required arguments.
- Examples
    ```bash
    python evaluate_baselines.py --evaluate_id 'gpt2-milu-full-sys-background(0)' \
                                 --nlg_name 'gpt2' \
                                 --gpt2_checkpoint_dname "act_resp.4" \
                                 --nlu_name 'milu' \
                                 --nlu_model_name 'full-sys' \
                                 --apply_noise 'True' \
                                 --noise_type 'background(0)'
    ```
The results will be output to `outputs/baselines/<run_id>`.