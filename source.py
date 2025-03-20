import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Đọc tệp âm thanh
sr, stereo = wavfile.read("f.wav")

# Vẽ tín hiệu âm thanh
plt.figure(figsize=(10, 4))
plt.plot(stereo[:, 0], label="Waveform")
plt.xlabel("length of signal")
plt.ylabel("How a big!")
plt.title("Audio Signal")
plt.legend()
plt.show()
# sr, stereo = wavfile.read("f.wav")
print(f"Sampling Rate: {sr}")
print(f"Shape of Audio Data: {stereo.shape}")
sd.play(stereo, sr)
sd.wait()  # Đợi phát xong
with open("hz.txt", "w") as f:
    f.write(str(sr))
with open("kernel.txt", "r") as f:
    w = np.array([int(num) for num in f.readline().split()])

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

