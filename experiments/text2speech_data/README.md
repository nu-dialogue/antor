# Text to speech MultiWOZ data
Convert text data in MultiWOZ to voice data using Google Cloud Text-to-Speech.

## 1. Setup Google Cloud Text-to-Speech API
1. Follow official procedure and obtrain a secret key
    - https://cloud.google.com/text-to-speech

2. Save the secret key in a directory of your choice

## 2. Convert data
Specify the path of the secret key.
```bash
python generate.py --gcp_secret_key_fpath ./secret_key.json
```
The result voice data will be output to `outputs/multiwoz/<part>`.