import pytest
import re
from pytorch_mums.mums import MUMS

# Consider: @pytest.mark.filterwarnings('ignore::UserWarning')
#   https://docs.pytest.org/en/7.1.x/how-to/capture-warnings.html#pytest-mark-filterwarnings
#   Or as pytest.ini: [pytest] filterwarnings = ignore::UserWarning

@pytest.fixture
def json_data():
    dataset = MUMS(root='/import/c4dm-datasets/MUMS/')
    return dataset.json_data

def test_pitch_height_str(json_data):
    all_pitch_height_str = [value['pitch_height_str'] for value in json_data.values() if value['type_str'] == 'note']
    all_pitch_height_str = set(filter(None, all_pitch_height_str))  # filter None values
    assert all(re.match(r"[A-G][b#]?\d", pitch_height_str) for pitch_height_str in all_pitch_height_str)

def test_pitch_class_str(json_data):
    all_pitch_class_str = [value['pitch_class_str'] for value in json_data.values() if value['type_str'] == 'note']
    all_pitch_class_str += [value['root_pitch_class_str'] for value in json_data.values() if value['type_str'] == 'chord']
    all_pitch_class_str = set(filter(None, all_pitch_class_str))  # filter None values
    assert all(re.match(r"[A-G][b#]?", pitch_class_str) for pitch_class_str in all_pitch_class_str)

def test_pitch(json_data):
    all_pitch = [value['pitch'] for value in json_data.values() if value['type_str'] == 'note']
    all_pitch = set(filter(None, all_pitch))  # filter None values
    assert all(isinstance(pitch, int) for pitch in all_pitch)
    assert all(0 <= pitch <= 127 for pitch in all_pitch)

def test_accordion_chord_quality_str(json_data):
    all_chord_quality_str = [value['chord_quality_str'] for value in json_data.values() if value['type_str'] == 'chord' and value['instrument_name_str'] == 'accordion']
    all_chord_quality_str = set(filter(None, all_chord_quality_str))  # filter None values
    assert all(re.match(r"^(DIM 7TH|DOM 7TH|MAJOR|MINOR)$", chord_quality_str) for chord_quality_str in all_chord_quality_str)

def test_guitar_chord_quality_str(json_data):
    all_chord_quality_str = [value['chord_quality_str'] for value in json_data.values() if value['type_str'] == 'chord' and value['instrument_name_str'] == 'guitar']
    all_chord_quality_str = set(filter(None, all_chord_quality_str))  # filter None values
    assert all(re.match(r"^(DOMINANT NINTH|DOMINANT SEVENTH|ELEVENTH|FLAT 7 SHARP 9|MAJOR SEVENTH|MAJOR SIXTH|MINOR SEVENTH|NINTH|FIFTH)$", chord_quality_str) for chord_quality_str in all_chord_quality_str)