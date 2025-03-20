import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
from time import sleep 
sys.stdout.reconfigure(encoding='utf-8')
sr, stereo = wavfile.read("f.wav")
print("Thành công đọc file audio!")

print("phát âm thanh")
sd.play(stereo, sr) 
print("hiển thị sinal lên biểu đồ")
plt.figure(figsize=(10, 4))
plt.plot(stereo[:, 0], label="Waveform")
plt.xlabel("length of signal")
plt.ylabel("How a big!")
plt.title("Audio Signal")
plt.legend()
plt.show()

sd.wait()  # Đợi phát xong

print("lưu tần số audio")
with open("hz.txt", "w") as f:
    f.write(str(sr))

sleep(1)
print("đọc kernel")
with open("kernel.txt", "r") as f:
    w = np.array([int(num) for num in f.readline().split()])

def clearing(x, file):
    with open(file, "w") as f: 
        def process(x):
            y = np.convolve(x, w, mode='full')
            f.write(" ".join(map(str, y)) + "\n")
        maxSub = 10
        idx = 0
        while idx * maxSub <= len(x):
            process(x[idx * maxSub: min(idx * maxSub + maxSub, len(x))])
            idx += 1
sleep(5)
print("chúng ta các tín hiệu thành các tín hiệu nhỏ có độ dài là 10, rồi lần lượt lưu vào từng hàng vào file")
clearing(np.array(stereo[:, 1]), file="chanel2.txt")
clearing(np.array(stereo[:, 0]), file="chanel1.txt")

sleep(1)
print("lúc này chúng ta sẽ có 2 file chanel1 và chanel 2 và 1 file lưu tần số là hz")


