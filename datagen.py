import argparse
import logging
import os
import sys

# C:\Users\PRANAV REDDY\Desktop\micmon\utils
# exec(open(r"C:\Users\PRANAV REDDY\Desktop\micmon\audio\directory.py").read())
from micmon.audio.segment import AudioSegment
from micmon.audio.directory import AudioDirectory
from micmon.audio.file import AudioFile
# exec(open("micmon/audio/directory.py").read())
# exec(open("micmon/audio/file.py").read())
# exec(open("/home/pi/micmon/audio/file.py").read())
# exec(open("micmon/audio/segment.py").read())
# exec(open("/home/pi/micmon/audio/segment.py").read())
# exec(open("micmon/dataset/writer.py").read())
# exec(open("/home/pi/micmon/dataset/writer.py").read())
from micmon.dataset.writer import DatasetWriter

logger = logging.getLogger(__name__)
defaults = {
    'sample_duration': 1,
    'sample_rate': 6000,
    'channels': 1,
    'ffmpeg_bin': "C:/ffmpeg/bin/ffmpeg",
}


def create_dataset(audiodir1: str, datasetdir1: str,
                   low_freq: int = AudioSegment.default_low_freq,
                   high_freq: int = AudioSegment.default_high_freq,
                   bins: int = AudioSegment.default_bins,
                   sample_duration: float = defaults['sample_duration'],
                   sample_rate: int = defaults['sample_rate'],
                   channels: int = defaults['channels'],
                   ffmpeg_bin: str = defaults['ffmpeg_bin']):

    audiodir1 = os.path.abspath(audiodir1)
    datasetdir1 = os.path.abspath(datasetdir1)

    print(f"audio dir : {audiodir1} \n dataset dir : {datasetdir1}")

    audio_dirs = AudioDirectory.scan(audiodir1)
    # print(os.path.expanduser(ffmpeg_bin))

    for audiodir1 in audio_dirs:
        dataset_file = os.path.join(datasetdir1, os.path.basename(audiodir1.path) + '.npz')
        print(f"dataset file : {dataset_file}")
        logger.info(f'Processing audio sample {audiodir1.path}')

        with AudioFile(audiodir1.audio_file, audiodir1.labels_file,
                       sample_duration=sample_duration, sample_rate=sample_rate, channels=channels,
                       ffmpeg_bin=os.path.expanduser(ffmpeg_bin)) as reader, \
                DatasetWriter(dataset_file, low_freq=low_freq, high_freq=high_freq, bins=bins) as writer:
            for sample in reader:
                writer += sample


def main():
    # noinspection PyTypeChecker
    audiodir1 = "datasets/sound-detect/audio"  # "~/dataset/sound-detect/audio"
    datasetdir1 = "datasets/sound-detect/data"  # ~/dataset/sound-detect/data"

    parser = argparse.ArgumentParser(
        description='''
Tool to create numpy dataset files with audio spectrum data from a set of labelled raw audio files.''',

        epilog='''
- audiodir1 should contain a list of sub-directories, each of which represents a labelled audio sample.
  audiodir1 should have the following structure:

  audiodir1/
    -> train_sample_1
      -> audio.mp3
      -> labels.json
    -> train_sample_2
      -> audio.mp3
      -> labels.json
  ...

- labels.json is a key-value JSON file that contains the labels for each audio segment. Example:

   {
     "00:00": "negative",
     "02:13": "positive",
     "04:57": "negative",
     "15:41": "positive",
     "18:24": "negative"
   }

  Each entry indicates that all the audio samples between the specified timestamp and the next entry or
  the end of the audio file should be applied the specified label.

- datasetdir1 is the directory where the generated labelled spectrum dataset in .npz format will be saved.
  Each dataset file will be named like its associated audio samples directory.''',

        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--audiodir1', default=audiodir1,
                        help='Directory containing the raw audio samples directories to be scanned.')
    parser.add_argument('--datasetdir1', default=datasetdir1,
                        help='Destination directory for the compressed .npz files containing the '
                             'frequency spectrum datasets.')
    parser.add_argument('--low', help='Specify the lowest frequency to be considered in the generated frequency '
                                      'spectrum. Default: 20 Hz (lowest possible frequency audible to a human ear).',
                        required=False, default=AudioSegment.default_low_freq, dest='low_freq', type=int)

    parser.add_argument('--high', help='Specify the highest frequency to be considered in the generated frequency '
                                       'spectrum. Default: 20 kHz (highest possible frequency audible to a human ear).',
                        required=False, default=AudioSegment.default_high_freq, dest='high_freq', type=int)

    parser.add_argument('-b', '--bins', help=f'Specify the number of frequency bins to be used for the spectrum '
                                             f'analysis (default: {AudioSegment.default_bins})',
                        required=False, default=AudioSegment.default_bins, dest='bins', type=int)

    parser.add_argument('-d', '--sample-duration', help=f'The script will calculate the spectrum of audio segments of '
                                                        f'this specified length in seconds (default: '
                                                        f'{defaults["sample_duration"]}).',
                        required=False, default=defaults['sample_duration'], dest='sample_duration', type=float)

    parser.add_argument('-r', '--sample-rate', help=f'Audio sample rate (default: {defaults["sample_rate"]} Hz)',
                        required=False, default=defaults['sample_rate'], dest='sample_rate', type=int)

    parser.add_argument('-c', '--channels', help=f'Number of destination audio channels (default: '
                                                 f'{defaults["channels"]})',
                        required=False, default=defaults['channels'], dest='channels', type=int)

    parser.add_argument('--ffmpeg', help=f'Absolute path to the ffmpeg executable (default: {defaults["ffmpeg_bin"]})',
                        required=False, default=defaults['ffmpeg_bin'], dest='ffmpeg_bin', type=str)

    opts, args = parser.parse_known_args(sys.argv[1:])
    return create_dataset(audiodir1=opts.audiodir1, datasetdir1=opts.datasetdir1, low_freq=opts.low_freq,
                          high_freq=opts.high_freq, bins=opts.bins, sample_duration=opts.sample_duration,
                          sample_rate=opts.sample_rate, channels=opts.channels, ffmpeg_bin=opts.ffmpeg_bin)


if __name__ == '__main__':
    main()
