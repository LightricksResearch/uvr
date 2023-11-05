import onnxruntime as ort
from .ModelData import *
import librosa
import torch
from .separate import SeperateMDX
from pydub import AudioSegment
from pathlib import Path

if torch.cuda.is_available():
    device, run_type = torch.device('cuda:0'), ['CUDAExecutionProvider']
else:
    device, run_type = torch.device('cpu'), ['CPUExecutionProvider']

default_model_data = ModelData('UVR-MDX-NET Inst HQ 1', MDX_ARCH_TYPE)
ort_inferencer = ort.InferenceSession(default_model_data.model_path, providers=run_type)


class AudioUtils:

    def __init__(self):
        pass

    @staticmethod
    def split(audio_path, export_path='', model: ModelData = default_model_data):
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
        seperator.seperate(ort_inferencer)
        torch.cuda.empty_cache()

    @staticmethod
    def merge(first_audio, second_audio, output_path):
        audio1 = AudioSegment.from_file(first_audio)
        audio2 = AudioSegment.from_file(second_audio)
        merged_audio = audio1.overlay(audio2)
        merged_audio.export(output_path, format='wav')

# if __name__ == "__main__":
#     base_dir = Path(__file__).parent.resolve()
#     test_output =  base_dir/ "output"
#     input_path =  base_dir / "flowers_short.wav"
#     vocal_output_path = test_output / f"{input_path.stem}_Vocals.wav"
#     test_output.mkdir(exist_ok=True)
#     AudioUtils.split(str(input_path), str(test_output))
#     instrumental_output_path = test_output / f"{input_path.stem}_Instrumental.wav"
#     AudioUtils.merge(vocal_output_path, instrumental_output_path,  test_output/ "")


    