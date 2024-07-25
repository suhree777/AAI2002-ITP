import os
import json

def create_mood_folders(base_directory, moods):
    """ Create a directory for each mood. """
    mood_directories = {}
    for mood in moods:
        mood_path = os.path.join(base_directory, mood)
        os.makedirs(mood_path, exist_ok=True)
        mood_directories[mood] = mood_path
    return mood_directories

def process_and_save_json_files(source_directory, mood_directories, num_samples):
    """ Process each JSON file and save it in the corresponding mood directory. """
    for filename in os.listdir(source_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(source_directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            mood = data['mood']
            if mood in mood_directories:
                # Truncate sequences
                for instrument, sequence in data['instruments'].items():
                    data['instruments'][instrument] = sequence[:num_samples]
                
                # Save the modified data to the corresponding mood folder
                target_file_path = os.path.join(mood_directories[mood], filename)
                with open(target_file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f"Processed {filename} and saved to {target_file_path}")

directory_path = 'ym2413_project_bt/3_processed_freq/'
sample_path = 'ym2413_project_bt/sample_seed/'
summary_path = os.path.join(directory_path, 'summary.json')
with open(summary_path, 'r') as f:
    data = json.load(f)
    instrument_list = data['top_instrument_names']
    moods = data['mood_labels']
    sample_rate = data['desired_sample_rate']
    size_multiplyer = data['size_multiplyer']

print(f"Summary file list has been loaded")
print(f"Desired sample rate: {sample_rate}")
print(f"Mood labels: {moods}")
print(f"Instrument list of {len(instrument_list)}: {instrument_list}")

mood_directories = create_mood_folders(sample_path, moods)
source_dir = 'ym2413_project_bt/3_processed_freq/data'
source_directory = os.path.join(directory_path, 'data')  # Assuming the JSON files are here
process_and_save_json_files(source_directory, mood_directories, (sample_rate * size_multiplyer))
