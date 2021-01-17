
import numpy as np
import matplotlib.pyplot as plt

print('Enter the text: ', end='')
s = input()
l = len(s)

fig = plt.figure(figsize=(2*l,1))
ax = fig.add_axes((0,0,1,1))
ax.axis('off')
ax.text(0,0,s, weight='bold', size=100)
fig.savefig('%s.png'%s)
plt.close(fig)

N = 1000
arr = plt.imread('%s.png'%s)[::-1, :, 0].T
xs = np.random.randint(0, arr.shape[0], (l*N,))
ys = np.random.randint(0, arr.shape[1], (l*N,))
X = np.vstack([xs, ys]).T
mask = (arr[xs, ys] < 0.5)
X = X[mask]

d = 100
xs = np.random.randn(d)
ys = np.random.randn(d)
xs = xs / np.linalg.norm(xs)
ys = ys - ys.dot(xs)*xs 
ys = ys / np.linalg.norm(ys)
U = np.vstack([xs,ys])
data = X.dot(U)

np.savetxt("hidden_text.csv", data, delimiter=",")
