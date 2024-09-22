from pathlib import Path

CHANNELS_NUMBER = 1
FRAME_RATE = 4000
SAMPLE_WIDTH = 2
FRAGMENT_LENGTH_MS = 2000
DIRECTORY = Path.cwd() / "data"
EXTENSIONS = [
    "*wav", "*aiff", "*pcm", "*mp3", "*aac", "*ac3", "*ogg vorbis", "*opus", 
    "*amr", "*wma", "*mp2", "*flac", "*alac", "*wavpack", "*tta", "*dts", 
    "*speex", "*vqf", "*realaudio", "*mpc"
]
BYPASS_BAD_VALUES = True