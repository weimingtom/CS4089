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

def get_sub_directories(GivenDIR) :
    a = os.listdir(GivenDIR)
    b = ""
    for i in a :
        current_iterator = GivenDIR + os.path.sep + i + os.path.sep
        if os.path.isdir(current_iterator) :
            b = b + current_iterator + " "
    return b

def save_training_model(GDir) :
    requiredDir = get_sub_directories(GDir)
    os.system("./speaker-recognition.py -t enroll -i \"" + requiredDir + "\" -m " + GDir + os.path.sep + TRAINING_DIR + ".model")

def usage_instructions() :
    print("python main.py " + "<test/train> " + "[<name of individual>]")
    sys.exit(1)

def predict_trained_model(TRAINING_DIR) :
    os.chdir(TRAINING_DIR)
    get_audio_input(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, "PREDICT.wav")
    os.chdir("..")
    os.system("./speaker-recognition.py --t predict -i \"" + TRAINING_DIR + os.path.sep + "PREDICT.wav" + "\" -m " + TRAINING_DIR + os.path.sep + TRAINING_DIR + ".model")
    os.system("rm " + TRAINING_DIR + os.path.sep + "PREDICT.wav")

if __name__ == "__main__" :
    ## Audio Record Globals --> START
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    ## Audio Record Globals <-- END
    if len(sys.argv) < 2 :
        usage_instructions()
    TRAINING_DIR = "srecogout"
    # take command line arguments
    type_of_method = str(sys.argv[1])
    # simple condition check
    if type_of_method == "train" :
        username = str(sys.argv[2])
        requiredDir = TRAINING_DIR + os.path.sep + username
        value = os.path.isdir(requiredDir)
        if not value :
            os.makedirs(requiredDir)
        save_training_data(requiredDir)
        save_training_model(TRAINING_DIR)
    elif type_of_method == "test" :
        predict_trained_model(TRAINING_DIR)
    elif type_of_method == "genTrainModel" :
        save_training_model(TRAINING_DIR)
    elif type_of_method == "recordAudio" :
        username = str(sys.argv[2])
        requiredDir = TRAINING_DIR + os.path.sep + username
        value = os.path.isdir(requiredDir)
        if not value :
            os.makedirs(requiredDir)
        save_training_data(requiredDir)
    else :
        usage_instructions()
