import torch
import torch.utils.data as data
import pandas as pd
import os
import re
from .utils import str2midi
from warnings import warn
from sklearn.preprocessing import LabelEncoder
import scipy.io.wavfile

class MUMS(data.Dataset):
    """ PyTorch dataset for MUMS.
        Inspired by pytorch-nsynth: https://github.com/kwon-young/pytorch-nsynth
    
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

    def __init__(self, root, transform=None, target_transform=None,
                 include_dirs=[],
                 blacklist_pattern=[],
                 categorical_field_list=['instrument_name_str', 'instrument_family_str']):
        
        assert(isinstance(root, str))
        assert(isinstance(include_dirs, list))
        assert(isinstance(blacklist_pattern, list))

        self.root = root
        self.include_dirs = include_dirs

        PATH_CSV = './src/dirs.csv'  # csv file listing bottom-level directories
        df_directories = pd.read_csv(PATH_CSV)
        
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
        self.le = []
        for i, field in enumerate(self.categorical_field_list):
            self.le.append(LabelEncoder())
            field_values = [value[field] for value in self.json_data.values()]
            self.le[i].fit(field_values)

        self.transform = transform
        self.target_transform = target_transform
        return
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, list, dict]:
        # TODO: audio files aren't same length so need to adjust duration somewhere - transforms arg?
        # https://iver56.github.io/audiomentations/waveform_transforms/adjust_duration/
        name = self.filenames[idx]
        _, sample = scipy.io.wavfile.read(name)
        target = self.json_data[os.path.splitext(os.path.basename(name))[0]]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.le)]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return [sample, *categorical_target, target]

if __name__ == "__main__":
    pass
    # import yaml

    # with open('../../cosi/config/mums.yaml') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # mums_dataset = MUMS(root=config['path']['mums'],
    #     include_dirs=config['include_dirs'],
    #     blacklist_pattern=config['blacklist_pattern'])

    # mums_dataset.json_data

    # # audio samples are loaded as an int16 numpy array
    # # rescale intensity range as float [-1, 1]
    # toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # # use instrument_family and instrument_source as classification targets
    # dataset = NSynth(
    #     "../nsynth-test",
    #     transform=toFloat,
    #     blacklist_pattern=["string"],  # blacklist string instrument
    #     categorical_field_list=["instrument_family", "instrument_source"])
    # loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # for samples, instrument_family_target, instrument_source_target, targets \
    #         in loader:
    #     print(samples.shape, instrument_family_target.shape,
    #           instrument_source_target.shape)
    #     print(torch.min(samples), torch.max(samples))



