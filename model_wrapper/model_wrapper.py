from BeatNet.BeatNet import BeatNet
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import torch
import time
from audio_midi import PianoTranscription, load_audio, sample_rate
import midi_score

#In order to run this, download the pretrained models for PM2S via:
# wget https://zenodo.org/record/7429669/files/_model_state_dicts.zip
# unzip _model_state_dicts.zip

class MuscribeModelWrapper:
    """
    Wrapper class to get midi, process key signatures, etc. etc. 

    beat_model: a FUNCTION that takes in a set of midi notes, and returns beat onset probability
    time_model: a FUNCTION that takes in a set of midi notes, and returns time signature probability
    key_model: a FUNCTION that takes in a set of midi notes, and returns key predictions
    """
    def __init__(
        self,
        beat_model: any = BeatNet(model=1),
        time_model: any = None,
        key_model: any = midi_score.RNNKeySignatureProcessor(),
        device: str =  "cuda" if torch.cuda.is_available() else "cpu",
        use_custom_model = False
    ) -> None:
        self.beat_model = beat_model
        self.time_model = time_model
        self.key_model = key_model
        self.device = device
        self.use_custom_model = use_custom_model

    def get_midi(self, audio_path, output_midi_path):
        (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)
        transcriptor = PianoTranscription(device=self.device)
        transcribe_time = time.time()
        transcriptor.transcribe(audio, output_midi_path)
        print("Transcribe time: {:.3f} s".format(time.time() - transcribe_time))
        if output_midi_path:
            return midi_score.read_note_sequence(output_midi_path)
        
    def get_beats(self, use_midi = False, path = None, midi_notes = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if(self.use_custom_model == False):
            beatnet = BeatNet(model=1)
            beat_activ = beatnet.activation_extractor_online(path)
        else:
            pred = self.beat_model(midi_notes).squeeze(0)
            beat_activ =  torch.zeros(pred.shape[0], 2, device = device)
            beat_activ[pred[0:] > 0/5 ,0] = 1
            beat_activ[pred[1:] > 0/5 ,1] = 1
        db_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        return db_tracker(beat_activ)  # Using DBN offline inference to infer beat/downbeats

    def get_keysig(self, midi_notes):
        key_model = midi_score.RNNKeySignatureProcessor()
        return key_model.process(midi_notes)

    def get_hand_parts(self, midi_notes):
        hand_model = midi_score.RNNHandPartProcessor()
        return hand_model.process(midi_notes)