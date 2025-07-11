from torch import Tensor, from_numpy
from torch.utils.data import Dataset
import pandas as pd
import os
import json
import re
from .datautils.transform import str2midi
from .datautils.encode import ConditionalLabelEncoder
from warnings import warn
import soundfile as sf
from typing import Sequence, Optional, Callable

SRC_DIR = os.path.dirname(__file__)  # ..
REGISTRY_CSV = os.path.join(SRC_DIR, 'registry.csv')  # lists base directories for audio files


class MUMS(Dataset):
    """ PyTorch dataset for MUMS.
        Adapted from pytorch-nsynth:
            https://github.com/kwon-young/pytorch-nsynth

    Args:
        data_path: Root directory of dataset.
        json_path: Path to the data file in JSON format. If it exists,
            it will be used to load the data file. Otherwise, the data file will be
            generated and saved to this path. If None, the data file will be 
            generated "on demand" in memory and not saved.
        split: TODO Dataset split to use. Currently not implemented.
        transform: A sequence of transforms to apply to the audio samples.
            Each transform should be a callable that takes in a sample
            and returns a transformed version.
        include_dirs: List of directories to include in the dataset.
            If empty, all directories are included. Refer to registry.csv.
        blacklist_pattern: List of strings used to blacklist dataset elements.
            If one of the strings is present in the audio filename, this sample
            is removed from the dataset and data file. Case-insensitive.
    """

    SCHEMA = {
    'shared': ['instrument_name_str', 'instrument_family_str', 'instrument_source_str'],
    'by_type': {
        'note': ['pitch_str', 'pitch_class_str'],
        'chord': ['root_pitch_class_str', 'chord_quality_str']
        },
    }

    def __init__(self, 
                 data_path: str,
                 json_path: Optional[str] = None,
                 split: Optional[str] = None,  # TODO: train, val, test
                 transform: Optional[Sequence[Callable]] = None,  # TODO: sequence
                 include_dirs: Sequence[str] = [],
                 blacklist_pattern: Sequence[str] = [],
                 ):

        self.data_path = data_path
        self.json_path = json_path
        self.split = split
        self.transform = transform

        if self.json_path is not None:
            if not self.json_path.endswith('.json'):
                self.json_path += '.json'

            if os.path.exists(self.json_path):
                print(f"json file found at path. Loading data file from {self.json_path}...")
                warn("Ignoring args to include_dirs and blacklist_pattern. "
                     "Use collate_fn instead to curate in DataLoader, or regenerate data file.")
                self.data = json.load(open(self.json_path, 'r'))
                self.filenames = list(self.data.keys())
            else:
                print(f"json file not found at path {self.json_path}. Generating data file and saving to path...")
                self._generate_datafile(include_dirs, blacklist_pattern)

        else:
            print("json_path is None. Data file will not be saved. Generating data \"file\" in memory...")
            self._generate_datafile(include_dirs, blacklist_pattern)

    # TODO: split logic
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    def _generate_datafile(self, 
                           include_dirs: Sequence[str],
                           blacklist_pattern: Sequence[str]):
        # TODO: setup logging, set destination for logs (dir/data.log)
        # see comments after generation below

        df_directories = pd.read_csv(REGISTRY_CSV)

        if include_dirs:  # otherwise include all directories by default
            df_directories = \
                df_directories[df_directories['subpath'].isin(include_dirs)]

        self.filenames = []
        self.data = {}

        blacklist = lambda x: any(re.search(pattern, x, re.IGNORECASE)
                              for pattern in blacklist_pattern)

        # TODO: optimise via multiprocessing, prompt for num_workers?

        for index, row in df_directories.iterrows():
            base_dir = os.path.join(self.data_path, row['subpath'])

            if blacklist(base_dir):
                continue

            instrument_name_str = row['instrument_name']
            instrument_family_str = row['instrument_family']
            instrument_source_str = row['instrument_source']
            type_str = row['type']

            for f in os.listdir(base_dir):
                if f.endswith('.wav'):
                    if blacklist(f):
                        continue

                    path_f = os.path.join(base_dir, f)
                    self.filenames.append(path_f)

                    targets_f = {'instrument_name_str': instrument_name_str,
                                 'instrument_family_str': instrument_family_str,
                                 'instrument_source_str': instrument_source_str,
                                 'type_str': type_str}

                    if type_str == 'note':
                        pitch_str = re.search(r'[A-Ga-g]#?\d', f)

                        if pitch_str is not None:
                            pitch_str = pitch_str.group(0).upper()
                            pitch_class_str = re.search(r'[A-Ga-g]#?', pitch_str).group(0)
                            pitch = str2midi(pitch_str)

                        else:
                            # TODO: change warn to log
                            warn(f"Pitch (class) not found in or extracted from {f}")
                            pitch_str = None
                            pitch = None
                            pitch_class_str = None

                        targets_f['pitch_str'] = pitch_str
                        targets_f['pitch_class_str'] = pitch_class_str
                        targets_f['pitch'] = pitch

                    elif type_str == 'chord':
                        if 'ELECTRIC GUITAR' in path_f:
                            pitch_class_str = re.search(r'_[A-Ga-g]#?', f).group(0)
                            targets_f['root_pitch_class_str'] = pitch_class_str[1:]  # remove underscore

                            chord_quality_str = re.search(r'GUITAR( |_)[A-Z](( ?(([A-Z0-9]+)?))+)?', base_dir).group(0)[7:]   # remove leading 'GUITAR'
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
                            # TODO: change warn to log
                            warn(f"No pitch class for {f}")
                            targets_f['root_pitch_class_str'] = None
                            warn(f"No chord quality for {f}")
                            targets_f['chord_quality_str'] = None

                    self.data[path_f] = targets_f

        encoder = ConditionalLabelEncoder(self.SCHEMA, exclude=['pitch_str'])
        encoder.fit(self.data)
        encoder.transform(self.data)
        print("DEBUG: Data file generated with the following keys:")
        print(self.data.keys())
        print("DEBUG: Data file contains the following (first five) items:")
        print(list(self.data.items())[:5])  # print first 5 items

        print(f"Generated data file for {len(self.filenames)} files")

        if self.json_path is not None:
            print(f"Saving data file to {self.json_path}...")
            with open(self.json_path, 'w') as f:
                json.dump(self.data, f, indent=4)

            # TODO: write data.log, include_dirs.txt, blacklist_pattern.txt and
            # save output (labels.json?) from ConditionalLabelEncoder to dir
            # using json_path as basename (i..e without json suffix)

            print(f"All done!")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> tuple[Tensor, list, dict[str, str | int]]:
        # TODO: rewrite in accordance with sol
        name = self.filenames[idx]
        sample, sr = sf.read(name)  # TODO: torchaudio.load
        sample = from_numpy(sample)
        target = self.data[name]
        # TODO: get encoded labels from self.data
        if self.transform is not None:
            sample = self.transform(sample)
        return [sample, target]
