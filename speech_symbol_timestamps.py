import whisper_timestamped
import json

def audio_query_json(audio_path, save_to_file=True, json_output_path="speech_symbol_timestamps.json"):
    # Load the desired Whisper model
    model = whisper_timestamped.load_model("base", device="cpu")
    
    # Transcribe the audio file
    result = whisper_timestamped.transcribe(model, audio_path, language="ja")
    
    # Print the complete transcription
    print("Complete transcription:", result["text"])
    
    # Function to distribute the time of a word among its symbols
    def distribute_time_equally(start, end, text, decimals=4):
        duration = end - start
        num_symbols = len(text)
        duration_per_symbol = duration / num_symbols
        symbols_times = []
        for i in range(num_symbols):
            symbol_start = start + i * duration_per_symbol
            symbol_end = start + (i + 1) * duration_per_symbol
            symbols_times.append({
                "text": text[i],
                "start": round(symbol_start, decimals),
                "end": round(symbol_end, decimals)
            })
        return symbols_times
    
    # Prepare the data to be saved in JSON
    data_to_save = {
        "transcription": result["text"],
        "segments": []
    }
    
    for segment in result["segments"]:
        segment_info = {
            "segment_id": segment['id'],
            "start": segment['start'],
            "end": segment['end'],
            "symbols": []
        }
        for word in segment.get("words", []):
            symbols_times = distribute_time_equally(word['start'], word['end'], word['text'])
            segment_info["symbols"].extend(symbols_times)
        data_to_save["segments"].append(segment_info)
    
    # Save the data in a JSON file if save_to_file is True
    if save_to_file:
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)
        print(f"Results saved in: {json_output_path}")
    
    return data_to_save

# Example of use
audio_path = "/home/andromeda/freelancer/AudioPhoneticsLab/test_audios/japanes_1.mp3"
transcription_json = audio_query_json(audio_path, save_to_file=True)
