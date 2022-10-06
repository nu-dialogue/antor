# Speech Recognition Error Simulation
Simulate speech recognition errors using original text data in MultiWOZ and transcribed text data.

You can directly use the confusion matrices with SNR settings used in our paper (`outputs/multiwoz/background(<SNR>).zip`). 
Simply extract each zip file in its original directory as shown below:
```bash
unzip "outputs/multiwoz/background(0).zip" -d outputs/multiwoz/
```

If you wish to create new confusion matrix yourself, please follow the steps below.

## Build Confusion Marix
Specify the path of the secret key and the type of noise used in `noisy_speech_data` and `speech2text_data`.
```bash
python build_confusion_matrix.py
```
The result will be output to `outputs/multiwoz/<noise_type>/<part>`.