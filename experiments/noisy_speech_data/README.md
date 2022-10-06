# Noisy Speech Data
## 1. Prepare resources (Exmpales)
- Background Noise
    - [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50#download)
        ```bash
        cd resources/background_noise/
        git clone https://github.com/karolpiczak/ESC-50.git
        python prepare_audio.py
        ```

- Impulse Responses
    - [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)
        ```bash
        cd resources/impulse_responses/
        wget --no-check-certificate https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip
        python prepare_audio.py
        ```
        - In prepare_audio.py, we resamples origianl audio data for applying noise and speech recognision.
            - Reference
                - http://makotomurakami.com/blog/2020/06/08/5522/
                - https://www.wizard-notes.com/entry/python/soundfile

## 2. Apply noise to speech data
Specify the type of original audio data (part, side) and the noise type. For the list of available noise type, see `core.py`.
```bash
python apply_noise.py --noise_type "background(0)"
```
The result noisy voice data will be output to `outputs/multiwoz/<noise_type>/<part>`.