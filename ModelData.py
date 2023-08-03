import os
import json
import hashlib
from .gui_data.constants import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')


def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''

    with open(dictionary) as d:
        data = d.read()

    return json.loads(data)


class ModelData:
    def __init__(self, model_name: str,
                 is_primary_model_primary_stem_only=False,
                 is_primary_model_secondary_stem_only=False):

        self.is_gpu_conversion = -1
        self.is_normalization = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_denoise = False
        self.mdx_batch_size = 1
        self.is_mdx_ckpt = False
        self.wav_type_set = 'PCM_16'
        self.mp3_bit_set = '320k'
        self.save_format = 'WAV'
        self.is_invert_spec = False
        self.is_mixer_mode = False
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = MDX_ARCH_TYPE
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_model_primary_stem = None
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = False
        self.is_dry_check = False
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.all_models = []
        self.is_secondary_model_activated = False
        self.is_secondary_model = False
        self.secondary_model = None
        self.secondary_model_scale = None
        self.is_ensemble_mode = False

        self.margin = 44100
        self.chunks = 0
        self.get_mdx_model_path()
        self.get_model_hash()
        if self.model_hash:
            mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
            self.model_data = self.get_model_data(MDX_HASH_DIR, mdx_hash_MAPPER)
            if self.model_data:
                self.compensate = self.model_data["compensate"]
                self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                self.primary_stem = self.model_data["primary_stem"]  # 'Instrumental'
                self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]  # 'Vocals

        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        self.pre_proc_model_activated = False

        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

    def get_mdx_model_path(self):

        if self.model_name.endswith(CKPT):
            # self.chunks = 0
            # self.is_mdx_batch_mode = True
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)
        for file_name, chosen_mdx_model in mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")

        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")

    def get_model_data(self, model_hash_dir, hash_mapper):

        model_settings_json = os.path.join(model_hash_dir, "{}.json".format(self.model_hash))

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

    def get_model_hash(self):
        try:
            with open(self.model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                self.model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            self.model_hash = hashlib.md5(open(self.model_path, 'rb').read()).hexdigest()
