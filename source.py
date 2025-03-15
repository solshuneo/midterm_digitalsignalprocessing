import numpy as np
from scipy.io import wavfile
import sounddevice as sd

sr, stereo = wavfile.read("audio.wav")
sd.play(stereo, sr)
sd.wait()  # Đợi phát xong
with open("hz.txt", "w") as f:
    f.write(str(sr))
with open("kernel.txt", "r") as f:
    w = np.array([float(num) for num in f.readline().split()])

def clearing(x, file):
    with open(file, "w") as f: 
        def process(x):
            # print(x)
            # pass
            y = np.convolve(x, w, mode='full')

            f.write(" ".join(map(str, y)) + "\n")
        maxSub = 10
        idx = 0
        while idx * maxSub <= len(x):
            process(x[idx * maxSub: min(idx * maxSub + maxSub, len(x))])
            idx += 1
clearing(np.array(stereo[:, 1]), file="chanel2.txt")
clearing(np.array(stereo[:, 0]), file="chanel1.txt")

# [ 0  0  0  0  1 -2  3 -4  3  0]
# [ 0  0  0  0 -1  2 -2  2 -3  3]

