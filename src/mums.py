from torch import Tensor, from_numpy
import torch.utils.data as data
import pandas as pd
import os
import re
from .utils import str2midi
from warnings import warn
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
from typing import Sequence, Optional, Any
from pathlib import Path

class MUMS(data.Dataset):
    """ PyTorch dataset for MUMS.
        Adapted from pytorch-nsynth: https://github.com/kwon-young/pytorch-nsynth
    
    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        include_dirs (list): List of bottom-level directories to include in the dataset.
            If empty, all bottom-level directories are included. Refer to directories.csv.
        blacklist_pattern (list): List of strings used to blacklist dataset elements.
            If one of the strings is present in the audio filename, this sample
            together with its metadata is removed from the dataset. Case-insensitive.
    """

    def __init__(self, root : str,
                 include_dirs : Sequence[str] = [], blacklist_pattern : Sequence[str] = [],
                 transform : Optional[callable] = None, target_transform : Optional[callable] = None,
                 categorical_field_list : Sequence[str] = ['instrument_name_str', 'instrument_family_str']
                 ):

        self.root = root
        self.include_dirs = include_dirs

        META_CSV = Path('./src/dirs.csv')  # csv file listing (mostly) bottom-level directories
        df_directories = pd.read_csv(META_CSV)
        
        if self.include_dirs:    # otherwise include all directories by default
            df_directories = df_directories[df_directories['subpath'].isin(self.include_dirs)]
    
        self.filenames = []
        self.json_data = {} # metadata

        blacklist = lambda x: any(re.search(pattern, x, re.IGNORECASE) for pattern in blacklist_pattern)

        for index, row in df_directories.iterrows():
            path_dir = os.path.join(self.root, row['subpath'])

            if blacklist(path_dir):
                continue

            instrument_name_str = row['instrument_name']
            instrument_family_str = row['instrument_family']
            instrument_source_str = row['instrument_source']
            type_str = row['type']

            for f in os.listdir(path_dir):
                if f.endswith('.wav'):
                    if blacklist(f):
                        continue

                    path_f = os.path.join(path_dir, f)
                    self.filenames.append(path_f)

                    targets_f = {'instrument_name_str': instrument_name_str,
                                  'instrument_family_str': instrument_family_str,
                                  'instrument_source_str': instrument_source_str,
                                  'type_str': type_str}

                    match type_str:
                        case 'note':
                            pitch_height_str = re.search(r'[A-Ga-g]#?\d', f)

                            if pitch_height_str is not None:
                                pitch_height_str = pitch_height_str.group(0).upper()
                                pitch_class_str = re.search(r'[A-Ga-g]#?', pitch_height_str).group(0)
                                pitch = str2midi(pitch_height_str)

                            else:
                                warn(f"Pitch height not found in or extracted from {f}")
                                pitch_height_str = None
                                warn(f"Pitch class not found in or extracted from {f}")
                                pitch_class_str = None
                                warn(f"Pitch not found in or extracted from {f}")
                                pitch = None
                                
                            targets_f['pitch_height_str'] = pitch_height_str
                            targets_f['pitch_class_str'] = pitch_class_str
                            targets_f['pitch'] = pitch

                        case 'chord':
                            if 'ELECTRIC GUITAR' in path_f:
                                pitch_class_str = re.search(r'_[A-Ga-g]#?', f).group(0)
                                targets_f['root_pitch_class_str'] = pitch_class_str[1:]  # remove underscore

                                chord_quality_str = re.search(r'GUITAR( |_)[A-Z](( ?(([A-Z0-9]+)?))+)?', path_dir).group(0)[7:]   # remove leading 'GUITAR'
                                if re.match(r'[A-Z](( ?(([A-Z0-9]+)?))+)?S$', chord_quality_str):
                                    chord_quality_str = chord_quality_str[:-1]  # remove trailing 'S'
                                elif re.match(r'[A-Z](( ?(([A-Z0-9]+)?))+)? STOPPED', chord_quality_str):
                                    chord_quality_str = chord_quality_str[:-8]  # remove trailing ' STOPPED'

                                targets_f['chord_quality_str'] = chord_quality_str  # default case

                            elif 'ACCORDION' in path_f:
                                pitch_class_str = re.search(r' [A-Ga-g]#? [A-Z]+', f).group(0)
                                if 'FLAT' in pitch_class_str:
                                    targets_f['root_pitch_class_str'] = f'{pitch_class_str[1:2]}b'  # use b symbol
                                    chord_quality_str = re.search(r' [A-Ga-g]#? [A-Z]+ ?([A-Z]+)? ?([A-Z0-9]+)?', f).group(0)[8:]   # remove leading pitch char and 'FLAT'

                                elif '#' in pitch_class_str:
                                    targets_f['root_pitch_class_str'] = pitch_class_str[1:3]  # remove trailing word
                                    chord_quality_str = re.search(r' [A-Ga-g]#? [A-Z]+ ?([A-Z]+)? ?([A-Z0-9]+)?', f).group(0)[4:]   # remove leading pitch char and '#'
                                else:
                                    targets_f['root_pitch_class_str'] = pitch_class_str[1:2]  # remove trailing word
                                    chord_quality_str = re.search(r' [A-Ga-g]#? [A-Z]+ ?([A-Z]+)? ?([A-Z0-9]+)?', f).group(0)[3:]   # remove leading pitch char

                                targets_f['chord_quality_str'] = chord_quality_str

                            elif 'ORGAN' in path_f:
                                warn(f"No pitch class for {f}")
                                targets_f['root_pitch_class_str'] = None
                                warn(f"No chord quality for {f}")
                                targets_f['chord_quality_str'] = None

                    self.json_data[path_f] = targets_f

        self.categorical_field_list = categorical_field_list
        self.label_encoders = []  # NB: renamed from self.le in pytorch-nsynth  
        for i, field in enumerate(self.categorical_field_list):
            self.label_encoders.append(LabelEncoder())
            field_values = [value[field] for value in self.json_data.values()]
            self.label_encoders[i].fit(field_values)

        self.transform = transform
        self.target_transform = target_transform
        return
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx) -> tuple[Tensor, list, dict[str, Any]]:
        # TODO: audio files aren't same length so need to adjust duration somewhere - transforms arg?
        #       https://iver56.github.io/audiomentations/waveform_transforms/adjust_duration/
        name = self.filenames[idx]
        sample, sr = sf.read(name)
        sample = from_numpy(sample)
        target = self.json_data[name]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.label_encoders)]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return [sample, *categorical_target, target]



