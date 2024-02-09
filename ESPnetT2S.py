# Standard library imports
import json
import os

# Third-party imports
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from scipy.io.wavfile import write
import whisper_timestamped

# Local application imports
from auxiliar_functions_for_audio_query import distribute_time_equally, add_consonant_vowel_info, calculate_pitch



class ESPnetTextToByte:
    """
    Processes an audio file using the provided ESPnet model and configuration.

    Parameters:
    - espnet_model (object): An instance of the ESPnet model for audio processing.
    - config_file (str): The path to the configuration file associated with the ESPnet model.
    - audio_file (str): The path to the input audio file for processing.

    Returns:
    - bytes: The processed audio data as a byte object.

    Raises:
    - Any exceptions raised during the audio processing with the ESPnet model.

    Note:
    - This function takes an ESPnet model, a configuration file, and an text file as input.
    - It returns the processed audio data as a byte object.
    """
    def __init__(self):
        self.text2speech = None
    def build(self, model, config, vocoder_tag, device="cpu"):
        """
        Builds the text-to-speech (TTS) model for audio processing.

        Parameters:
        - model (str): Path to the TTS model file.
        - config (str): Path to the TTS model configuration file.
        - vocoder_tag (str): Tag associated with the vocoder.
        - device (str): Device on which the TTS model will be loaded (default is "cpu").

        Raises:
        - Exception: If an error occurs during the initialization of the TTS model.

        Note:
        - This method initializes the text2speech attribute using the provided model, config, vocoder_tag, and device.
        """
        try:
            # Initialize the Text2Speech instance from ESPnet
            self.text2speech = Text2Speech.from_pretrained(
                model_file=str_or_none(model),
                train_config=str_or_none(config),
                vocoder_tag=str_or_none(vocoder_tag),
                device=device
            )
        except Exception as e:
            # Raise an exception if an error occurs during TTS model initialization
            raise e
            
    def remove_new_line(self, text_file_path):
        """
        Removes new lines from the text in the specified file.

        Parameters:
        - text_file_path (str): Path to the text file.

        Returns:
        - str: Text content without new lines.

        Note:
        - This function reads the content of the text file at the specified path and removes new line characters.
        """
        # Open the text file, read its content, and replace new lines with an empty string
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
        return text
            
    def get_wav_data(self, text_file_path):
        """
        Generates WAV data from the text content of the specified file.

        Parameters:
        - text_file_path (str): Path to the text file.

        Returns:
        - numpy.ndarray: WAV data as a one-dimensional NumPy array.

        Note:
        - This function utilizes the `remove_new_line` method to preprocess the text and then generates WAV data
          using the text-to-speech (TTS) model.
        """
        # Remove new lines from the text content
        text = self.remove_new_line(text_file_path)

        # Generate WAV data using the text-to-speech model
        with torch.no_grad():
            wav = self.text2speech(text)["wav"]

        # Convert WAV data to a one-dimensional NumPy array
        wavdata = wav.view(-1).cpu().numpy()
        return wavdata
    
    def get_byte_data(self, text_file, output_path="audio_byte_file.raw"):
        """
        Converts WAV data generated from the text content into a byte file.

        Parameters:
        - text_file (str): Path to the text file.
        - output_path (str): Path to the output byte file (default is "audio_byte_file.raw").

        Note:
        - This function utilizes the `get_wav_data` method to generate WAV data from the text content.
        - The WAV data is then written to a binary file in raw byte format.
        """
        # Generate WAV data from the text content
        wavdata = self.get_wav_data(text_file)

        # Write the WAV data to a binary file in raw byte format
        with open(output_path, 'wb') as file:
            file.write(wavdata.tobytes())
            
    def get_audio(self, text_file):
        """
        Generates an audio file from the text content.

        Parameters:
        - text_file (str): Path to the text file.

        Note:
        - This function is for just test purpose.
        - This function utilizes the `get_wav_data` method to generate WAV data from the text content.
        - The WAV data is then used to create an audio file in WAV format.
        """
        # Generate WAV data from the text content
        wavdata = self.get_wav_data(text_file)

        # Get the sample rate from the text-to-speech model
        samplerate = self.text2speech.fs

        # Write the WAV data to an audio file in WAV format
        write("audio.wav", samplerate, wavdata)

    def audio_query_json(self, audio_path, save_to_file=True, json_output_path="speech_symbol_timestamps.json", mapping_file="files/mapping.json"):
        """
        Transcribes an audio file to text, enriches each transcribed word with detailed phonetic information 
        (consonants and vowels), calculates the pitch for each symbol, and identifies interrogative sentences.
        
        Parameters:
        - audio_path (str): The path to the audio file for transcription.
        - save_to_file (bool): Whether to save the output to a JSON file. Defaults to False.
        - json_output_path (str): Path where the JSON output will be saved if save_to_file is True.
        - mapping_file (str): Path to the JSON file containing mappings for consonant and vowel information.
        
        Returns:
        - dict: A dictionary containing the complete transcription, word details including phonetic information, 
                pitch, and whether each word forms a question, along with some metadata about the audio processing.
        """
        # Load the model and transcribe the audio
        model = whisper_timestamped.load_model("small", device=self.device)
        result = whisper_timestamped.transcribe(model, audio_path, language="ja")
        print("Complete transcription:", result["text"])

        # Initialize the main dictionary to store the transcription and word details
        audio_query_data = {
            "transcription": result["text"],
            "words": []
        }

        # Process each word in the transcription
        for segment in result["segments"]:
            for word in segment.get("words", []):
                # Distribute time equally among the symbols of the word
                symbols_times = distribute_time_equally(word['start'], word['end'], word['text'])

                # Add consonant and vowel information to each symbol
                symbols_times = add_consonant_vowel_info(symbols_times, mapping_file)
                
                # Calculate and add pitch information to each symbol
                symbols_times = calculate_pitch(audio_path, symbols_times)

                # Create a dictionary for each word with its details
                word_detail = {
                    "symbols": symbols_times,
                    "is_interrogative": "か" in word['text'] or "?" in word['text'],  # Check if the word is interrogative
                    "complete_word": word['text']
                }
                
                # Add the detailed word to the list
                audio_query_data["words"].append(word_detail)

        # Add additional metadata related to the audio processing
        metadata = {
            "speedScale": None,
            "pitchScale": None,
            "intonationScale": None,
            "volumeScale": None,
            "prePhonemeLength": None,
            "postPhonemeLength": None,
            "outputSamplingRate": 24000,  # Set the output sampling rate explicitly
            "outputStereo": None,
            "kana": result["text"]  # The transcribed text in Kana
        }

        # Update the main dictionary with the metadata
        audio_query_data.update(metadata)

        # Save the results to a file if requested
        if save_to_file:
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(audio_query_data, json_file, ensure_ascii=False, indent=4)
            print(f"Results saved in: {json_output_path}")
        
        return audio_query_data



if __name__ == "__main__":
    
    # File paths for the pre-trained model, configuration, and input text
    model_path = "model/train.total_count.ave_10best.pth"
    config_file_path = "model/config.yaml"
    text_file_path = "text.txt"

    # Vocoder tag specifying the type of vocoder to be used
    # its a voice encoder. it works by separate the carrier signal (which typically represents the spectral content of the voice) and the modulator signal (which represents the characteristics such as pitch and intensity).
    vocoder_tag = "none" # we can use vocoder_tag Parallel WaveGAN, and MelGAN  

    # Device on which the model will be loaded (default is "cpu")
    device = "cpu"  # Change to "cuda" for GPU acceleration if a compatible GPU is available

    # Initialize the espnet model
    espnet = ESPnetTextToByte()
    espnet.build(model_path, config_file_path, vocoder_tag, device)

    # Initialize output file path and and getbyte data
    output_path = "audio_byte_file.raw"
    espnet.get_byte_data(text_file_path, output_path)
    
    # Generate audio from text file
    espnet.get_audio(text_file_path)

    # Generate the audio-query JSON file
    audio_path = "japanesef32.wav"
    espnet.audio_query_json(audio_path)