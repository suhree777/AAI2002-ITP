## Download Datasets
1. [NES-MDB Dataset](https://drive.google.com/file/d/1Z6RtPpKWCTYUJjvRV3PhyYuEDzdT82r4/view)<br />
   **Language Modeling Format (155 MB)** <br />
   This format is specifically designed for tasks that involve sequence modelling, such as training a Transformer model. Given that the LakhNES project involves generating music based on learned patterns from a dataset, the Language Modeling Format is the most suitable. It structures the data in a way that is optimized for models that learn to predict subsequent events in a sequence, which is exactly what you need for music generation.

2. [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)<br />
   **LMD-full (The full collection of 176,581 deduped MIDI files)** <br />
   For pre-training, you want the broadest possible diversity and quantity of data to give the model a solid foundation in music from various genres, not just chiptune or video game music. LMD-full offers the largest collection of MIDI files, providing the model with a wide variety of musical structures and patterns. Although it contains some corrupt files, the sheer volume and diversity of the dataset make it the best choice for pre-training before fine-tuning on the more specific NES-MDB dataset.

   ### Unzip Datasets
   a. Using **Command Line (cmd)** > Right click and **Run as administrator** <br /><br />
   bi. Type <code>cd</code> to change the current working directory to the location of the .tar.gz file for unzipping. <br />
   **Extract .tar.gz file to the current working directory** <br />
   <code>tar -xvzf nesmdb_nlm.tar.gz</code> <br />
   <code>tar -xvzf lmd_full.tar.gz</code> <br /><br />
   bii. **OR Extract .tar.gz file from source path to destination path** <br />
   Type <code>tar</code> to specify a source and destination file path. <br />
   <code>tar -xvzf C:\PATH\TO\SOURCE\filename.tar.gz -C C:\PATH\TO\DESTINATION</code>
   
   ## getdata.bat
   This batch script is to download and prepare various language modelling datasets -- WikiText-2, WikiText-103, enwik8, text8, and Penn Treebank.
   
