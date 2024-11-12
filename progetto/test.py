from mgeval import core, utils

if __name__ == "__main__":
    midi_song = midi.read_midifile("scarlatti_8_bars/real/000000000001.mid")
    print(midi_song[1])
    f = core.extract_feature("scarlatti_8_bars/real/000000000001.mid")
    metric = ("note_length_hist")
    ev = getattr(core.metrics(), metric)
    score = ev(f)
    print(metric + ": " + str(score))
