# Simulator NLU Joint BERT
A script to adjust the amount of data when training Joint BERT, based on the original script (`ConvLab-2/convlab2/nlu/jointBERT`).

# Usage
## 1. Data Preprocess
```
python preprocess.py [data_id; Arbitrary string to identify the data to be created)] \
                     [mode; Speaker of trainig data to be used (all/usr/sys)] \
                     --vocab_level_by_turn_dpath [The absolute path of preprocessed vocabulary level information] \
                     --vocab_level_tolerance [The list of vocabulary levels to be included in training data. See `experiments/vocabulary_level/readme.md` for more detailed information]
```
- For example:
    ```bash
    python preprocess.py cefrjA1 \
                         sys \
                         --vocab_level_by_turn_dpath /home/ohashi/dev/antor/experiments/vocabulary_level/outputs/cefrj \
                         --vocab_level_tolerance NonAlpha A1
    ```
    - The preprocessed data will be output in `./multiwoz_data/[data_id]-[mode]/`.
- You don't need to specify `--vocab_level_by_turn_dpath` and `--vocab_level_tolerance` if you don't limit the training data at vocabulary levels.

## 2. Prepare config file
Recommended config file path: `./configs/<config file name>`
- For example:
    ```json
    {
        "data_dir": ".../user_nlu/multiwoz_data/cefrjA1-sys", # absolute path
        "output_dir": ".../user_nlu/output/cefrjA1-sys", # absolute path
        "log_dir": ".../user_nlu/log/cefrjA1-sys", # absolute path
        "zipped_model_path": ".../user_nlu/output/cefrjA1-sys/cefrjA1-sys.zip", # absolute path
        "DEVICE": "cuda:0",
        "seed": 42,
        ...
    }
    ```

## 3. Train Model
1. Specify **absolute path** of ConvLab-2's training script on `./train.sh`.
    ```bash
    train_script_path=".../ConvLab-2/convlab2/nlu/jointBERT" # absolute path
    ```

2. Run script with config file's relative path.
    ```bash
    chmod 744 train.sh
    ./train.sh configs/cefrjA1-sys.json # relative path
    ```

## (Option) Test Model
1. Specify **absolute path** of ConvLab-2's test script on `./test.sh`.
    ```bash
    train_script_path=".../ConvLab-2/convlab2/nlu/jointBERT" # absolute path
    ```
2. Run script with config file's **relative path**.
    ```bash
    $ chmod 744 test.sh
    (.venv)$ ./test.sh configs/cefrjA1-sys.json # relative path
    ```

## (Option) TensorBoard
1. Run script with log directory's **relative path**.
  ```bash
  tensorboard --logdir=log/cefrjA1-sys # relative path
  ```