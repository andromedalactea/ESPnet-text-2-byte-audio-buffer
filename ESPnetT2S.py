from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import torch
from scipy.io.wavfile import write
from speech_symbol_timestamps import audio_query_json
import whisper_timestamped
import json
import os

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

    def audio_query(self, audio_path, save_to_file=True, json_output_path="speech_symbol_timestamps.json"):
        """
        Transcribes an audio file to text using the Whisper model and generates a JSON
        file with the transcription and timestamps for each word and its symbols.

        Parameters:
        - audio_path (str): Path to the audio file to be transcribed.
        - save_to_file (bool, optional): Whether to save the transcription results to a JSON file. Defaults to True.
        - json_output_path (str, optional): Path where the JSON file with transcription results will be saved. Defaults to "speech_symbol_timestamps.json".

        Returns:
        - JSON: A JSON containing the complete transcription and detailed segments with timestamps.

        The function performs the following steps:
        - Loads the Whisper model specified for transcription.
        - Transcribes the audio file, specifying Japanese as the language.
        - Distributes the time equally among the symbols of each transcribed word.
        - Prepares and optionally saves the transcription and timestamps in a structured JSON format.
        """

        # Check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: The audio file '{audio_path}' does not exist.")
            return  # Stops the function execution if the file does not exist

        # Load the desired Whisper model. Specify the model type and the device (CPU or GPU).
        model = whisper_timestamped.load_model("base", device=self.device)
        
        # Transcribe the audio file to text. The language parameter is set to Japanese ('ja').
        result = whisper_timestamped.transcribe(model, audio_path, language="ja")
        
        # Print the complete transcription to the console for immediate viewing.
        print("Complete transcription:", result["text"])
        
        # This nested function calculates the time duration for each symbol within a word.
        # It ensures that the timestamps are distributed evenly across all symbols.
        def distribute_time_equally(start, end, text, decimals=4):
            duration = end - start  # Calculate the total duration of the word.
            num_symbols = len(text)  # Count the number of symbols in the word.
            duration_per_symbol = duration / num_symbols  # Calculate duration per symbol.
            symbols_times = []  # Initialize a list to store the timestamps of each symbol.
            for i in range(num_symbols):
                # Calculate the start and end time for each symbol.
                symbol_start = start + i * duration_per_symbol
                symbol_end = start + (i + 1) * duration_per_symbol
                # Append the symbol's text and its start/end times to the list.
                symbols_times.append({
                    "text": text[i],
                    "start": round(symbol_start, decimals),
                    "end": round(symbol_end, decimals)
                })
            return symbols_times  # Return the list of timestamps for each symbol.
        
        # Initialize a dictionary to store the complete transcription and timestamps for each segment.
        data_to_save = {
            "transcription": result["text"],
            "segments": []
        }
        
        # Process each segment from the transcription results.
        for segment in result["segments"]:
            # Initialize a dictionary to store segment details and symbol timestamps.
            segment_info = {
                "segment_id": segment['id'],
                "start": segment['start'],
                "end": segment['end'],
                "symbols": []
            }
            # Process each word in the segment to distribute timestamps among its symbols.
            for word in segment.get("words", []):
                symbols_times = distribute_time_equally(word['start'], word['end'], word['text'])
                segment_info["symbols"].extend(symbols_times)  # Append symbol timestamps to the segment.
            data_to_save["segments"].append(segment_info)  # Append the segment information to the main data.
        
        # Check if the transcription results should be saved to a JSON file.
        if save_to_file:
            # Open the specified JSON file in write mode and save the transcription data.
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)
            print(f"Results saved in: {json_output_path}")  # Print the path to the saved JSON file.
        
        return data_to_save  # Return the transcription data as a JSON object.



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
    audio_path = "audio.wav"
    espnet.audio_query(audio_path)


