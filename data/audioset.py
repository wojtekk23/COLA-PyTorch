import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import yt_dlp as youtube_dl


def download_video(ytid: str, video_folder='videos'):
    ytopts = {
        'outtmpl': f'{video_folder}/{ytid}',
        # 'format': 'bestuaudio/best',
        # 'format': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]',
        'format': 'bestaudio[ext=m4a]/best[ext=m4a][filesize<10M]/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'noplaylist': True,
        'audioquality': 5
    }
    try:
        with youtube_dl.YoutubeDL(ytopts) as ydl:
            video_info = ydl.extract_info(f'http://www.youtube.com/watch?v={ytid}', download=True)
            video_path = ydl.prepare_filename(video_info)
    except Exception as e:
        print(e)
        return None

    return f'{video_path}'


def process_video(video_path, start, end, sampling_rate=16000):
    y, sr = librosa.load(video_path, offset=start, duration=end - start)
    segment = librosa.feature.melspectrogram(y, sr, hop_length=160, win_length=400, n_fft=1024, n_mels=64, fmin=60, fmax=7800)
    return segment


class Audioset(Dataset):
    def __init__(self, info, sample_len=96, sampling_rate=16000, video_folder='videos'):
        super(Audioset, self).__init__()
        # TODO: labels in the last column?
        df = pd.read_csv(info, skiprows=3, usecols=range(3), names=['YTID', 'start_seconds', 'end_seconds'], quotechar='"')
        self.ytids = df['YTID']
        self.start_seconds = df['start_seconds']
        self.end_seconds = df['end_seconds']
        self.sample_len = sample_len
        self.sr = sampling_rate

    def __getitem__(self, ix):
        ytid = self.ytids[ix]
        start = self.start_seconds[ix]
        end = self.end_seconds[ix]

        video_path = download_video(ytid)
        if video_path is None:
            return None, None, None
        audio = process_video(video_path, start, end, sampling_rate=self.sr)
        os.remove(video_path)

        anchor_begin, positive_begin = np.random.randint(0, audio.shape[1] - self.sample_len, size=2)
        anchor = audio[:, anchor_begin:anchor_begin + self.sample_len]
        positive = audio[:, positive_begin:positive_begin + self.sample_len]

        return video_path, anchor, positive

    def __len__(self):
        return len(self.ytids)
