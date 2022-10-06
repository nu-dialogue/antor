import os
from google.api_core import exceptions
from google.cloud import texttospeech
import time

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

class text2wav:
    def __init__(self, secret_key_fpath='./secret-key.json'):
        if not os.path.isfile(secret_key_fpath):
            raise FileNotFoundError
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = secret_key_fpath
        
        self.client = texttospeech.TextToSpeechClient()

    def run(self, text, output_fname):
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-C",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        synthesis_input = texttospeech.SynthesisInput(text=text)
        exception = None
        for i in range(10):
            try:
                response = self.client.synthesize_speech(
                    request={"input": synthesis_input,
                             "voice": voice,
                             "audio_config": audio_config}
                )
                break
            except exceptions.InternalServerError as e:
                logger.warn(f"google.api_core.exceptions.InternalServerError: {str(e)} "
                            f"Retry synthesize_speech in 10 sec ({i+1} times)")
                time.sleep(10)
        else:
            raise exception

        with open(f"{output_fname}.wav", "wb") as out:
            out.write(response.audio_content)
        return response