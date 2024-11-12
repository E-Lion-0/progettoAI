from mgeval import core, utils
import pretty_midi
import midi

if __name__ == "__main__":
    midi_song = midi.read_midifile("scarlatti_8_bars/real/000000000001.mid")
    pretty_midi_s = pretty_midi.PrettyMIDI("scarlatti_8_bars/real/000000000001.mid")
    print(pretty_midi_s.instruments)
    f = core.extract_feature("scarlatti_8_bars/real/000000000001.mid")
    metric = ("note_length_hist")
    ev = getattr(core.metrics(), metric)
    score = ev(f)
    print(metric + ": " + str(score))
