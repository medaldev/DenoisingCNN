import matplotlib.pyplot as plt
import numpy as np

n = list(range(10, 120 + 5, 5))
a = [9.30467, 9.31562, 9.31944, 9.32122, 9.32218, 9.32276, 9.32313, 9.32339, 9.32358, 9.32371, 9.32382, np.nan, 9.32396,
     np.nan, 9.32405, np.nan, 9.32412, np.nan, 9.32417, np.nan, 9.3242, np.nan, 9.32423]

a = np.array(a)
plt.figure(figsize=(15, 8))

plt.tight_layout()
plt.plot(n, a, '.', markersize=15, label='', color="b")
# plt.legend()
plt.xlabel("Размер сетки")
plt.ylabel("Yв")
plt.show()
