import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sys
from time import sleep 
sys.stdout.reconfigure(encoding='utf-8')
print("đọc tần số của audio")
with open("hz.txt", "r") as f:
    sr = int(f.read())

sleep(1)
print("đọc kernel")
with open("kernel.txt", "r") as f:
    w = np.array([int(num) for num in f.readline().split()])

def deConvolution(y, w):
    w_len = len(w)
    y_len = len(y)
    x_len = y_len - w_len + 1

    A = np.zeros((y_len, x_len))
    for i in range(y_len):
        for j in range(x_len):
            if 0 <= i - j < w_len:
                A[i, j] = w[i - j]
    x_recovered, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return x_recovered

stereo_audio_list = []
origin_audio_list = []
sleep(1)
print("chúng ta sẽ đọc dữ liệu dược từ 2 chanel được mã hóa")
print("sử dụng linear least squares để khôi phục")
print("và chúng ta sẽ không sử dụng linear least squares để xem xét sự khác biệt giữa cả 2")
with open("chanel1.txt", "r") as f1, open("chanel2.txt", "r") as f2:
    for line1, line2 in zip(f1, f2):
        chanel1 = np.array([int(num) for num in line1.split()])
        chanel2 = np.array([int(num) for num in line2.split()])
        stereo_recovered = np.column_stack((deConvolution(chanel1, w), deConvolution(chanel2, w)))
        origin_recovered = np.column_stack((chanel1, chanel2))
        stereo_audio_list.append(stereo_recovered)
        origin_audio_list.append(origin_recovered)


stereo_audio = np.vstack(stereo_audio_list)
origin_audio = np.vstack(origin_audio_list)
def plot_waveform(stereo_audio, sr=44100):
    time = np.linspace(0, len(stereo_audio) / sr, num=len(stereo_audio))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)
    plt.plot(time, stereo_audio[:, 0], color='blue', label='Channel 1 (Left)')
    plt.xlabel("length of signal")
    plt.ylabel("How a big!")
    plt.title("Audio Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

sleep(3)
print("Âm thanh sau khi nhận được và chuyển đổi")
sd.play(stereo_audio * 0.0001, samplerate=sr)
plot_waveform(origin_audio)
sleep(3)
print("Am thanh khi nhận được")
sd.play(origin_audio * 0.00001, samplerate=sr)
plot_waveform(stereo_audio)
sd.wait()  