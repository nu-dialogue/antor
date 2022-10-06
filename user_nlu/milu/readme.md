# Simulator NLU MILU
A script to adjust the amount of data when training Joint BERT, based on the original script (`ConvLab-2/convlab2/nlu/milu`).

# Usage
## 1. Prepare config file
- Recommended path of config file : `./configs/<config file name>`
  ```json
  "dataset_reader": {
      "type": "milu_resize",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        },
        "token_characters": {
          "type": "characters",
          "min_padding_length": 3
        }
      },
      "agent": "system", # user, system
      "vocab_level_by_turn_dpath": ".../experiments/vocabulary_level/outputs/cefrj", # The absolute path of preprocessed vocabulary level information
      "vocab_level_tolerance": ["NonAlpha+Stop", "A1"] # The list of vocabulary levels to be included in training data
    },
    "train_data_path": ".../ConvLab-2/data/multiwoz/train.json.zip", # absolute path
    "validation_data_path": ".../ConvLab-2/data/multiwoz/val.json.zip", # absolute path
    "test_data_path": ".../ConvLab-2/data/multiwoz/test.json.zip", # absolute path
    ...
  ```
- You don't need to specify `vocab_level_by_turn_dpath` and `--vocab_level_tolerance` if you don't limit the training data at vocabulary levels.
  - See `./configs/full-sys.jsonnet` for an example.

## 2. Train model
1. Run script with config file's **relative path** and outputs dir's **relative path**.
    ```bash
    (.venv)$ python train.py configs/cefrjA1-sys.jsonnet -s outputs/cefrjA1-sys/
    ```

## (Option) Evaluate model
1. Run script with model dir's **relative path**. See `evaluate.py` for more options.
    ```bash
    (.venv)$ python evaluate.py outputs/cefrjA1-sys
    ```

## (Option) TensorBoard
1. Run script with log directory's **relative path**.
  ```bash
  $ chmod 744 tensorboard.sh
  (.venv)$ ./tensorboard.sh outputs/cefrjA1-sys/ # relative path
  ```