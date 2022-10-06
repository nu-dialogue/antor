import os
import io
from google.api_core import exceptions
from google.cloud import speech
import time

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

class wav2text:
    def __init__(self, secret_key_fpath='./secret-key.json') -> None:
        if not os.path.isfile(secret_key_fpath):
            raise FileNotFoundError
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = secret_key_fpath

        self.client = speech.SpeechClient()

    def run(self, audio_fpath):  
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        with io.open(audio_fpath, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        response = self.client.recognize(config=config, audio=audio)
        return response