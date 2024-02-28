import matplotlib.pyplot as plt
import numpy as np

n = list(range(10, 70, 5))
a = [16.0674, 15.8476, 15.7758, 15.7434,
     15.726, 15.7156, 15.7089, 15.7043,
     15.701 ,15.6986, 15.6968, 15.6942,]

a = np.array(a)
plt.figure(figsize=(15, 8))

plt.tight_layout()
plt.plot(n, a, '.', markersize=15, label='', color="b")
# plt.legend()
plt.xlabel("Размер сетки")
plt.ylabel("Yв")
plt.show()
