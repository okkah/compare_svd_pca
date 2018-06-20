from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import os

# Import image
img = Image.open('jack.jpg')

# Grayscale
gray_img = img.convert('L')
gray_img.save('jack_gray.jpg')

# Define data
X = np.asarray(gray_img)
n, p = X.shape

# Standardize the data
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

# Perform SVD
U, s, V_svd = np.linalg.svd(X_std, full_matrices=True)
V_svd = V_svd.T
S = np.zeros((n, p))
S[:p, :p] = np.diag(s)

# Perform PCA
cov = X_std.T @ X_std / (n - 1)
W, V_pca = np.linalg.eig(cov)
# Sort eigenvectors with eigenvalues
index = W.argsort()[::-1]
W = W[index]
V_pca = V_pca[:, index]

# Low rank approximation
U2, s2, V_svd2 = np.linalg.svd(X, full_matrices=True)
ranks = [1, 5, 10, 20, 30, 50, 100, 600]
if not os.path.exists("./low_rank_approximation"):
    os.mkdir("./low_rank_approximation")
for rank in ranks:
    ur = U2[:, :rank]
    sr = np.matrix(linalg.diagsvd(s2[:rank], rank,rank))
    vr = V_svd2[:rank, :]
    b = np.asarray(ur*sr*vr)
    img2 = Image.fromarray(np.uint8(b))
    rank_padded = '%03d' % rank
    img2.save('./low_rank_approximation/jack_gray_svd_' + str(rank_padded) + '.jpg')

# Print results
print("Eigenvalues from SVD")
print(s ** 2 / (n - 1))
print("Engenvalues from PCA")
print(W)

print("V_svd")
print(V_svd)
print("V_pca")
print(V_pca)

# Plot results
plt.scatter(V_pca[:, 0], V_pca[:, 1], color='green', label='PCA')
plt.scatter(V_svd[:, 0], V_svd[:, 1], color='blue', label='SVD')
plt.scatter(X_std[:, 0], X_std[:, 1], color='red', label='Original')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper left')
plt.show()
