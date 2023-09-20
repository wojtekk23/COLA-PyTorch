import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import yt_dlp as youtube_dl
import ffmpeg


def download_video(ytid: str, start_sec: float, end_sec: float, video_folder='videos', quiet: bool = False):
    ytopts = {
        'outtmpl': f'{video_folder}/{ytid}.wav',
        # 'format': 'bestuaudio/best',
        # 'format': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]',
        'quiet': quiet,
        'format': 'bestaudio[ext=m4a]/best[ext=m4a][filesize<10M]/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'noplaylist': True,
        'audioquality': 5
    }
    try:
        with youtube_dl.YoutubeDL(ytopts) as ydl:
            video_info = ydl.extract_info(f'http://www.youtube.com/watch?v={ytid}', download=False)
            video_url = video_info['url']
            video_path = ydl.prepare_filename(video_info)
    except Exception as e:
        print(e)
        return None

    input_stream = ffmpeg.input(video_url, noaccurate_seek=None)
    output_stream = (
        input_stream
        .filter('atrim', start=start_sec, end=end_sec)
        .filter('asetpts', 'PTS-STARTPTS')
    )
    cmd = ffmpeg.output(output_stream, video_path, acodec='pcm_s16le').overwrite_output()
    cmd.run(quiet=quiet)

    return f'{video_path}'


def process_audio(video_path, sampling_rate=16000):
    y, sr = librosa.load(video_path, sr=sampling_rate)
    segment = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=160, win_length=400, n_fft=1024, n_mels=64, fmin=60, fmax=7800)
    return segment

def process_audio_stft(video_path, sampling_rate=16000):
    y, sr = librosa.load(video_path)
    segment = librosa.stft(n_fft=2048, win_length=2000, hop_length=500)
    pass


class Audioset(Dataset):
    def __init__(self, info, quiet=False, sample_len=96, sampling_rate=16000, video_folder='videos'):
        super(Audioset, self).__init__()
        # TODO: labels in the last column?
        df = pd.read_csv(info, skiprows=3, usecols=range(3), names=['YTID', 'start_seconds', 'end_seconds'], quotechar='"')
        self.ytids = df['YTID']
        self.start_seconds = df['start_seconds']
        self.end_seconds = df['end_seconds']
        self.sample_len = sample_len
        self.sr = sampling_rate
        self.quiet = quiet

    def __getitem__(self, ix):
        ytid = self.ytids[ix]
        start = self.start_seconds[ix]
        end = self.end_seconds[ix]

        video_path = download_video(ytid, start, end, quiet=self.quiet)
        if video_path is None:
            return None, None, None
        audio = process_audio(video_path, sampling_rate=self.sr)
        os.remove(video_path)

        anchor_begin, positive_begin = np.random.randint(0, audio.shape[1] - self.sample_len, size=2)
        anchor = audio[:, anchor_begin:anchor_begin + self.sample_len]
        positive = audio[:, positive_begin:positive_begin + self.sample_len]

        return video_path, anchor, positive

    def __len__(self):
        return len(self.ytids)


class LocalAudioset(Dataset):
    def __init__(self, audio_folder, sample_len=96, sampling_rate=16000):
        super(LocalAudioset, self).__init__()
        self.audio_paths = [os.path.join(audio_folder, filename) for filename in os.listdir(audio_folder) if filename.endswith('.wav')]
        self.sample_len = sample_len
        self.sr = sampling_rate

    def __getitem__(self, ix):
        audio_path = self.audio_paths[ix]

        audio = process_audio(audio_path, sampling_rate=self.sr)
        # If the audio clip is too short, pad it with zeros
        if audio.shape[1] < self.sample_len:
            padding = self.sample_len - audio.shape[1] + 50
            audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')

        try:
            anchor_begin, positive_begin = np.random.randint(0, audio.shape[1] - self.sample_len, size=2)
            anchor = audio[:, anchor_begin:anchor_begin + self.sample_len]
            positive = audio[:, positive_begin:positive_begin + self.sample_len]
        except Exception as e:
            print(audio.shape[1], self.sample_len, audio_path)
            raise e
        return audio_path, anchor, positive

    def __len__(self):
        return len(self.audio_paths)
