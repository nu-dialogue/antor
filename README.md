# ANTOR
This is the implementation of COLING 2022 paper:

Adaptive Natural Language Generation for Task-oriented Dialogue via Reinforcement Learning. [[arXiv]](https://arxiv.org/abs/2209.07873)

## Setup
Python == 3.7

1. Clone repository
    ```bash
    $ git clone --recursive git@github.com:nu-dialogue/antor.git
    $ cd antor
    ```
2. Install ConvLab-2
    ```bash
    cd ConvLab-2
    pip install -e .
    cd ../
    ```
3. Install antor
    ```bash
    pip install -e .
    python -m spacy download en_core_web_sm
    ```
    > It is ok to ignore pip's dependency error with ConvLab-2.


## Experiments
You can reproduce the three experiments performed in the paper by following these steps.

### 1. Prepare Models
Before running the experiment, the NLG and NLU models must be prepared respectively.
#### System NLG
- GPT-2 (Base model for reinforcement learning) and SC-GPT
    - Go to `sys_nlg/gpt2` and `sys_nlg/scgpt` and prepare each model following the `README.md` in the directory, respectively.
- SC-LSTM
    - (Since we use pre-trained models available in the ConvLab-2, there are no steps required.)
#### User NLU
- MILU and BERT NLU
    - Go to `user_nlu/milu` and `user_nlu/bert` and prepare the each model following the `README.md` in the directory, respectively.

### 2. Conduct experiments in each condition
#### 2-a In Clean Environment
1. Simply fine-tune the NLG (GPT-2) with NLU (BERT NLU or MILU)
    - Go to `experiments/ppo` and follow the `README.md` in the directory.
2. Evaluate the fine-tuned model
    - Go to `experiments/evaluate_model` and follow the `README.md` in the directory.

#### 2-b Conditions for Speech Recognition Error
1. Build the confusion matrix for ASR error simulation
    - Build the confusion matrix by performing the steps described in each `README.md` in `experiments/text2speech_data`, `experiments/noisy_speech_data`, `experiments/speech2text_data`, and `experiments/speech_error_simulation`, in that order.
    > You can skip this step by directly using the final confusion matrices we used in our paper. The usage can be found in `experiments/speech_error_simulation`.
    > In addition, we also publish the noisy transcribed text data used to build the confusion matrices. See `experiments/speech2text_data`.
2. Fine-tune NLG in ASR error simulation
    - Go to `experiments/ppo` and follow the `README.md` in the directory.
3. Evaluate the fine-tuned model
    - Go to `experiments/evaluate_model` and follow the `README.md` in the directory.

#### 2-c Conditions for Different Vocabulary Levels
1. Prepare vocabulary level
    - Go to `experiments/vocabulary_level` and follow the `README.md` in the directory.
2. Train NLU models with only certain vocabulary levels of data
    - Go to `user_nlu/milu` or `user_nlu/bert` and train each model following the `README.md` in the directory.
3. Fine-tune NLG
    - Go to `experiments/ppo` and follow the `README.md` in the directory.
4. Evaluate the fine-tuned model
    - Go to `experiments/evaluate_model` and follow the `README.md` in the directory.

## Citation
```
@article{ohashi2022adaptive,
  title={Adaptive Natural Language Generation for Task-oriented Dialogue via Reinforcement Learning},
  author={Ohashi, Atsumoto and Higashinaka, Ryuichiro},
  journal={arXiv preprint arXiv:2209.07873},
  year={2022}
}
```