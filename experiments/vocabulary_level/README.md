# Vocabulary restrictions by CEFR-J word list

## Vocabulary Levels
Assign one of six vocabulary levels to each utterance:
- A1: Utterances consisting of words included in the A1 vocabulary level of CEFR-J
- A2: Utterances consisting of words included in the A2 or A2 vocabulary level of CEFR-J
- B1: Utterances consisting of words included in the A1, A2, or B1 vocabulary level of CEFR-J
- B2: Utterances consisting of words included in the A1, A2, B1, or B2 vocabulary level of CEFR-J
- OOV: Utterances consisting  of words included in vocabulary levels higher than B2 level of CEFR-J
- NonAlpha+Stop: consisting only of non-alphabetic letters and stop-words

## Preprocessing
1. Prepare cefrj word list
    ```bash
    git clone https://github.com/openlanguageprofiles/olp-en-cefrj.git
    ```

2. Preprocess word list
    ```bash
    python preprocess.py
    ```
    - Turn-by-turn vocabulary level for each part (train, val, test) will be output to `outputs/<part>.json`
