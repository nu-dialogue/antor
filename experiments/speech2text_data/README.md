# Speech to text multiwoz data
Convert voice data in MultiWOZ to text data using Google Cloud Speech-to-Text.

The data with SNR settings used in our paper are already available in `outputs/multiwoz/background(<SNR>).zip`. 
Simply extract each zip file in its original directory as shown below:
```bash
unzip "outputs/multiwoz/background(0).zip" -d outputs/multiwoz/
```

If you wish to create new data yourself, please follow the steps below.

## 1. Setup Google Cloud Speech-to-Text API
1. Follow official procedure and obtrain a secret key
    - https://cloud.google.com/speech-to-text

2. Save the secret key in a directory of your choice
    > If you use the same secret key as in `text2speech_data`, this step is not necessary

## 2. Transcribe
Specify the path of the secret key and the type of noise used in `noisy_speech_data`.
```bash
python transcribe.py --noise_type "background(0)" \
                     --gcp_secret_key_fpath ./secret_key.json
```
The result text data will be output to `outputs/multiwoz/<noise_type>/<part>.json`.