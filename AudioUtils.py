from ModelData import *
import librosa
import torch
from separate import SeperateMDX
from pydub import AudioSegment


class AudioUtils:

    def __init__(self):
        pass

    @staticmethod
    def split(audio_path, export_path='', model: ModelData = ModelData('UVR-MDX-NET Inst HQ 1', MDX_ARCH_TYPE)):
        librosa.load(audio_path, duration=3, mono=False, sr=44100)
        audio_file_base = os.path.splitext(os.path.basename(audio_path))[0]
        process_data = {
            'model_data': model,
            'export_path': export_path,
            'audio_file_base': audio_file_base,
            'audio_file': audio_path,
            'set_progress_bar': lambda step, inference_iterations=1: None,
            'write_to_console': lambda progress_text, base_text="": None,
            'process_iteration': {},
            'cached_source_callback': lambda model_type, model_name: (None, None),
            'cached_model_source_holder': {},
            'list_all_models': [model.model_name],
            'is_ensemble_master': False,
            'is_4_stem_ensemble': False
        }

        seperator = SeperateMDX(model, process_data)
        seperator.seperate()
        torch.cuda.empty_cache()

    @staticmethod
    def merge(path1, path2):
        audio1 = AudioSegment.from_file(path1)
        audio2 = AudioSegment.from_file(path2)
        merged_audio = audio1.overlay(audio2)
        merged_audio.export('merged.wav', format='wav')
