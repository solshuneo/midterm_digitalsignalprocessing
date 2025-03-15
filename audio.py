import numpy as np
from scipy.io import wavfile
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

# 🔹 Đọc file kernel
with open("kernel.txt", "r") as f:
    w = np.array([float(num) for num in f.readline().split()])

# 🔹 Đọc file WAV (tránh lỗi chunk)
sr, stereo = wavfile.read("audio.wav", mmap=True)

# Nếu file có 2 kênh, tách riêng từng kênh
chanel1, chanel2 = stereo[:, 0], stereo[:, 1]

# 🔹 Thực hiện convolution
y1 = np.convolve(chanel1, w, mode='full')
y2 = np.convolve(chanel2, w, mode='full')

# 🔹 Dùng ma trận thưa (Sparse Matrix) để giảm bộ nhớ
def create_sparse_matrix(y_len, x_len, w):
    A = lil_matrix((y_len, x_len), dtype=np.float32)
    w_len = len(w)
    for i in range(y_len):
        for j in range(max(0, i - w_len + 1), min(x_len, i + 1)):
            A[i, j] = w[i - j]
    return A.tocsr()

# 🔹 Tạo ma trận thưa cho từng kênh
A1 = create_sparse_matrix(len(y1), len(chanel1), w)
A2 = create_sparse_matrix(len(y2), len(chanel2), w)

# 🔹 Giải hệ phương trình tuyến tính bằng phương pháp sparse LSQR
chanel1_recovered = lsqr(A1, y1)[0]
chanel2_recovered = lsqr(A2, y2)[0]

# 🔹 So sánh sai số
error = np.linalg.norm(chanel1 - chanel1_recovered)
print("\nSai số ||chanel1 - chanel1_recovered||:", error)
