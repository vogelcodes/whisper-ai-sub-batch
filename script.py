import os
from stable_whisper import load_model
from stable_whisper import results_to_sentence_srt
from stable_whisper import stabilize_timestamps

filesList = os.listdir('./')
string_array = ['.mp4', '.wav', '.flac', '.m4a', '.mp3']
mediaFiles = list(filter(lambda x: x[-4:] in string_array, filesList))
print('Files found: ' +  str(mediaFiles) )

# word-level 
# after you get results from modified model
# this treats a word timestamp as end time of the word
# and combines words if their timestamps overlap

models = ['large', 'medium', 'small', 'tiny']
i=2

# modified model should run just like the regular model but with additional hyperparameters and extra data in results
while True:
    try:
        if i > len(models):
            print("No more models to try. Increase your VRAM size.")
            break
        print('Trying model: '+ models[i])
        model = load_model(models[i])
        for file in mediaFiles:
            print('Transcribing file: '+ file)
            results = model.transcribe(file)
            stab_segments = results['segments']
            first_segment_word_timestamps = stab_segments[0]['whole_word_timestamps']

        # or to get token timestamps that adhere more to the top prediction
            stab_segments = stabilize_timestamps(results, top_focus=True)
            results_to_sentence_srt(results, file[:-4] +'.srt')
            print(results)

        break
    except Exception as e:
        print(e)
        i = i + 1