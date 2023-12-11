# Muscribe (Formerly Muscript)

**Note** Project is still under heavy development, a number of features might not work as intended.

This is a project meant to provide an end-to-end transcription from audio music files to sheet music. The project is largely divided into two parts:

- MP3 to MIDI: 
    - We are using the [piano transcription inference](https://github.com/qiuqiangkong/piano_transcription_inference) framework from ByteDance. Related paper: [High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times](https://arxiv.org/abs/2010.01815), 2020.
    - So far this framwork has a number of restrictions:
        - Can only handle piano music.
        - Resilience in face of noise / low-quality audio is unclear.
    - Although the framework works well under the given restrictions, we are exploring other solutions/expanding to include more functionalities.

- MIDI to MusicXML:
    - This part of the project is largely based around [PM2S](https://cheriell.github.io/research/PM2S/) (ISMIR 2022).
    - However, accuracy is lacking in beat tracking & note quantization, we're exploring newer models to enhance this part of the prediction.
    - PM2S also offers prediction models for hand tracking (piano) and time / key signature changes, which are still valuable. (Need to evaluate thoroughly.)

## Environments

To ensure compatibility, please use the provided conda environment specified in ```env.yaml```. For example, running the following will create a "muscribe" conda environment:

```cmd
$conda env create -f env.yaml
$conda activate muscribe
```

## Demo

For an end-to-end demo of the project, please see use the ```demo.ipynb``` notebook to train and use the model.
