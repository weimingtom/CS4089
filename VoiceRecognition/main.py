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

def save_training_data(requiredDir) :
    os.chdir(requiredDir)
    for i in range(0, 10) :
        j = str(i).zfill(2)
        print(j)
        get_audio_input(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, "FILE" + j + ".wav")
    os.chdir("../..")

def save_training_model(requiredDir, username) :
    os.system("./speaker-recognition.py --t enroll -i \"" + requiredDir + "\" -m " + requiredDir + "/../" +username + ".model")

def usage_instructions() :
    print("python main.py " + "<name of individual> " + "<test/train>")
    sys.exit(1)

def predict_trained_model(TRAINING_DIR, username) :
    os.chdir(TRAINING_DIR)
    get_audio_input(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, "PREDICT.wav")
    os.chdir("..")
    os.system("./speaker-recognition.py --t predict -i \"" + TRAINING_DIR + os.path.sep + "PREDICT.wav" + "\" -m " + TRAINING_DIR + os.path.sep + username + ".model")
    os.system("rm " + TRAINING_DIR + os.path.sep + "PREDICT.wav")

if __name__ == "__main__" :
    ## Audio Record Globals --> START
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    ## Audio Record Globals <-- END
    if len(sys.argv) != 3 :
        usage_instructions()
    TRAINING_DIR = "srecogout"
    username = str(sys.argv[1])
    # take command line arguments
    type_of_method = str(sys.argv[2])
    requiredDir = TRAINING_DIR + os.path.sep + username
    value = os.path.isdir(requiredDir)
    if not value :
        os.makedirs(requiredDir)
    # simple condition check
    if type_of_method == "train" :
        save_training_data(requiredDir)
        save_training_model(requiredDir, username)
    elif type_of_method == "test" :
        predict_trained_model(TRAINING_DIR, username)
    else :
        usage_instructions()
