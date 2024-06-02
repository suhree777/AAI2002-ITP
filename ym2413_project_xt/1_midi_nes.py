import itertools
import os
import random
import pretty_midi
import csv

# Define the pitch ranges for NES instruments
nes_instrument_name_to_min_pitch = {
    'p1': 33,
    'p2': 33,
    'tr': 21
}
nes_instrument_name_to_max_pitch = {
    'p1': 108,
    'p2': 108,
    'tr': 108
}

def is_instrument_monophonic(instrument):
    notes = instrument.notes
    last_note_start = -1
    for note in notes:
        assert note.start >= last_note_start
        last_note_start = note.start

    monophonic = True
    for i in range(len(notes) - 1):
        note0 = notes[i]
        note1 = notes[i + 1]
        if note0.end > note1.start:
            monophonic = False
            break
    return monophonic

def generate_nesmdb_midi_examples(
    midi_filepath,
    output_directory,
    min_num_instruments=1,
    min_length_seconds=5.,
    max_length_seconds=600.,
    filter_bad_times=True,
    min_pitch=21,
    max_pitch=108,
    filter_duplicates=True,
    include_drums=True,
    max_examples=16,
    max_duration_seconds=180.,
    emotion_mapping=None):

    midi_name = os.path.splitext(os.path.basename(midi_filepath))[0]

    if min_num_instruments <= 0:
        raise ValueError()

    if os.path.getsize(midi_filepath) > (512 * 1024): # 512K
        return

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_filepath)
    except:
        return

    midi_length = midi_data.get_end_time()
    if midi_length < min_length_seconds or midi_length > max_length_seconds:
        return

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if filter_bad_times:
                if note.start < 0 or note.end < 0 or note.end < note.start:
                    return
            note.start = round(note.start * 44100.) / 44100.
            note.end = round(note.end * 44100.) / 44100.

    instruments = midi_data.instruments
    drums = [i for i in instruments if i.is_drum]
    instruments = [i for i in instruments if not i.is_drum]

    instruments_in_range = []
    for instrument in instruments:
        pitches = [n.pitch for n in instrument.notes]
        min_pitch_instrument = min(pitches)
        max_pitch_instrument = max(pitches)
        if max_pitch_instrument >= min_pitch and min_pitch_instrument <= max_pitch:
            instruments_in_range.append(instrument)
    instruments = instruments_in_range
    if len(instruments) < min_num_instruments:
        return

    for instrument in instruments:
        instrument.notes = sorted(instrument.notes, key=lambda x: x.start)
    if include_drums:
        for instrument in drums:
            instrument.notes = sorted(instrument.notes, key=lambda x: x.start)

    instruments = [i for i in instruments if is_instrument_monophonic(i)]
    if len(instruments) < min_num_instruments:
        return

    if filter_duplicates:
        unique_notes = set()
        unique_instruments = []
        for instrument in instruments:
            pitches = ','.join(['{}:{:.1f}'.format(str(note.pitch), note.start) for note in instrument.notes])
            if pitches not in unique_notes:
                unique_instruments.append(instrument)
                unique_notes.add(pitches)
        instruments = unique_instruments
        if len(instruments) < min_num_instruments:
            return

    num_instruments = len(instruments)
    if num_instruments == 1:
        instrument_permutations = [(0, -1, -1), (-1, 0, -1), (-1, -1, 0)]
    elif num_instruments == 2:
        instrument_permutations = [(-1, 0, 1), (-1, 1, 0), (0, -1, 1), (0, 1, -1), (1, -1, 0), (1, 0, -1)]
    elif num_instruments > 32:
        instrument_permutations = list(itertools.permutations(random.sample(range(num_instruments), 32), 3))
    else:
        instrument_permutations = list(itertools.permutations(range(num_instruments), 3))

    if len(instrument_permutations) > max_examples:
        instrument_permutations = random.sample(instrument_permutations, max_examples)

    num_drums = len(drums) if include_drums else 0
    instrument_permutations_plus_drums = []
    for permutation in instrument_permutations:
        selection = -1 if num_drums == 0 else random.choice(range(num_drums))
        instrument_permutations_plus_drums.append(permutation + (selection,))
    instrument_permutations = instrument_permutations_plus_drums

    quarter_label = emotion_mapping.get(midi_name, 'Unknown')
    emotion_mapping_dict = {
        'Q1': 'happy',
        'Q2': 'angry',
        'Q3': 'sad',
        'Q4': 'relaxed'
    }
    emotion_label = emotion_mapping_dict.get(quarter_label, 'Other')
    output_subdir = f"{quarter_label}_{emotion_label}"
    file_output_directory = os.path.join(output_directory, output_subdir)
    os.makedirs(file_output_directory, exist_ok=True)

    for i, permutation in enumerate(instrument_permutations):
        lead1_program = pretty_midi.instrument_name_to_program('Lead 1 (square)')
        lead2_program = pretty_midi.instrument_name_to_program('Lead 2 (sawtooth)')
        bass_program = pretty_midi.instrument_name_to_program('Synth Bass 1')
        drum_program = pretty_midi.instrument_name_to_program('Breath Noise')
        lead1_instrument = pretty_midi.Instrument(program=lead1_program, name='p1', is_drum=False)
        lead2_instrument = pretty_midi.Instrument(program=lead2_program, name='p2', is_drum=False)
        bass_instrument = pretty_midi.Instrument(program=bass_program, name='tr', is_drum=False)
        drum_instrument = pretty_midi.Instrument(program=drum_program, name='no', is_drum=True)

        permutation_notes = []
        for midi_instrument_id, nes_instrument_name in zip(permutation, ['p1', 'p2', 'tr', 'no']):
            if midi_instrument_id < 0:
                permutation_notes.append(None)
            else:
                if nes_instrument_name == 'no':
                    midi_instrument = drums[midi_instrument_id]
                    valid_notes = midi_instrument.notes
                else:
                    midi_instrument = instruments[midi_instrument_id]
                    valid_notes = [n for n in midi_instrument.notes if n.pitch >= nes_instrument_name_to_min_pitch[nes_instrument_name] and n.pitch <= nes_instrument_name_to_max_pitch[nes_instrument_name]]
                permutation_notes.append(valid_notes)
        assert len(permutation_notes) == 4

        start_time = None
        end_time = None
        for notes in permutation_notes:
            if notes is None or len(notes) == 0:
                continue
            note_start = min([n.start for n in notes])
            note_end = max([n.end for n in notes])
            if start_time is None or note_start < start_time:
                start_time = note_start
            if end_time is None or note_end > end_time:
                end_time = note_end
        if start_time is None or end_time is None:
            continue

        if (end_time - start_time) > max_duration_seconds:
            end_time = start_time + max_duration_seconds

        for notes, instrument_name, instrument in zip(permutation_notes, ['p1', 'p2', 'tr', 'no'], [lead1_instrument, lead2_instrument, bass_instrument, drum_instrument]):
            if notes is None:
                continue

            if instrument_name == 'no':
                random_noise_mapping = [random.randint(1, 16) for _ in range(128)]

            last_note_end = -1
            for note in notes:
                velocity = note.velocity
                pitch = note.pitch
                note_start = note.start
                note_end = note.end

                if instrument_name == 'no' and note_start < last_note_end:
                    continue
                last_note_end = note_end

                assert note_start >= start_time
                if note_end > end_time:
                    continue
                assert note_end <= end_time

                velocity = 1 if instrument_name == 'tr' else int(round(1. + (14. * velocity / 127.)))
                assert velocity > 0
                if instrument_name == 'no':
                    pitch = random_noise_mapping[pitch]
                note_start = note_start - start_time
                note_end = note_end - start_time
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=note_start,
                        end=note_end
                    )
                )

        midi_output = pretty_midi.PrettyMIDI()
        for inst in [lead1_instrument, lead2_instrument, bass_instrument, drum_instrument]:
            if len(inst.notes) > 0:
                midi_output.instruments.append(inst)
        end_instrument = pretty_midi.Instrument(program=0, name='end', is_drum=False)
        end_instrument.notes.append(
            pretty_midi.Note(
                velocity=15,
                pitch=108,
                start=(end_time - start_time),
                end=(end_time - start_time) + .1
            )
        )
        midi_output.instruments.append(end_instrument)

        output_midi_filepath = os.path.join(file_output_directory, f"{midi_name}_{i}.mid")
        midi_output.write(output_midi_filepath)

def load_emotion_mapping(csv_filepath):
    emotion_mapping = {}
    with open(csv_filepath, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for row in reader:
            filename = os.path.splitext(os.path.basename(row[0]))[0]
            emotion_mapping[filename] = row[3]
    return emotion_mapping

if __name__ == '__main__':
    pretty_midi.pretty_midi.MAX_TICK = 1e16

    emotion_mapping = load_emotion_mapping('music_dataset/YM2413-MDB-v1.0.2/emotion_annotation/verified_annotation.csv')
    midi_data_directory = 'music_dataset/YM2413-MDB-v1.0.2/midi/adjust_tempo_remove_delayed_inst'
    output_directory = 'ym2413_project_xt/1_output'

    for midi_filename in os.listdir(midi_data_directory):
        if midi_filename.endswith('.mid'):
            generate_nesmdb_midi_examples(
                os.path.join(midi_data_directory, midi_filename),
                output_directory,
                emotion_mapping=emotion_mapping)
