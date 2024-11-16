import itertools as it

def str2midi(note_string : str) -> int:
  """
  Source: https://pythonhosted.org/audiolazy/_modules/audiolazy/lazy_midi.html#str2midi
  Given a note string name (e.g. "Bb4"), returns its MIDI pitch number.

  Args:
    note_string (str): Note string name, e.g. "Bb4".
  
  Returns:
    int: MIDI pitch number.
  """
  MIDI_A4 = 69
  data = note_string.strip().lower()
  name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
  accident2delta = {"b": -1, "#": 1, "x": 2}
  accidents = list(it.takewhile(lambda el: el in accident2delta, data[1:]))
  octave_delta = int(data[len(accidents) + 1:]) - 4
  return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
          )