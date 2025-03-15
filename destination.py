import numpy as np
import sounddevice as sd

with open("hz.txt", "r") as f:
    sr = int(f.read())
with open("kernel.txt", "r") as f:
    w = np.array([float(num) for num in f.readline().split()])

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

with open("chanel1.txt", "r") as f1, open("chanel2.txt", "r") as f2:
    for line1, line2 in zip(f1, f2):
        chanel1 = np.array([float(num) for num in line1.split()])
        chanel2 = np.array([float(num) for num in line2.split()])
        # np.set_printoptions(suppress=True)  # Tắt hiển thị số mũ
        # print(deConvolution(chanel1, w))
        # print(deConvolution(chanel2, w))
        stereo_recovered = np.column_stack((deConvolution(chanel1, w), deConvolution(chanel2, w)))
        # print(stereo_recovered)
        # break
        # 🔹 Cộng dồn vào danh sách
        stereo_audio_list.append(stereo_recovered)

stereo_audio = np.vstack(stereo_audio_list)

sd.play(stereo_audio, samplerate=sr)
sd.wait()  