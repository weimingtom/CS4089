import pyaudio
import wave
import os
import sys

def get_audio_input(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, WAVE_OUTPUT_FILENAME) :
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("start recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("stop recording")
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # print audio
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

if __name__ == "__main__" :
    # print("Hello World")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    TRAINING_DIR = "srecogout"
    username = str(sys.argv[1])
    requiredDir = TRAINING_DIR + os.path.sep + username
    value = os.path.isdir(requiredDir)
    if not value :
        os.makedirs(requiredDir)
    os.chdir(requiredDir)
    for i in range(0, 10) :
        j = str(i).zfill(2)
        print(j)
        get_audio_input(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, "FILE" + j + ".wav")
    os.chdir("../..")
