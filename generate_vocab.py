import os

def generate_vocabulary(input_dir, output_file):
    vocabulary = set()

    # Collect all unique events from the event sequence files
    for root, dirs, files in os.walk(input_dir):
        print(f"Processing folder: {os.path.basename(root)}")
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    events = f.read().strip().split()
                    vocabulary.update(events)

    # Convert the set to a list and sort it
    vocabulary = sorted(list(vocabulary))

    # Save the vocabulary to a file
    with open(output_file, 'w') as f:
        for event in vocabulary:
            f.write(event + '\n')

    print(f"Vocabulary generated with {len(vocabulary)} unique events.")


input_dir = 'event_sequences'
output_file = 'vocab.txt'
generate_vocabulary(input_dir, output_file)
