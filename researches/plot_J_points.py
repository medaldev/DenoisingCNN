import matplotlib.pyplot as plt
import numpy as np

n = list(range(10, 140, 5))

Uvych = np.array(
    [2.3724892363703756, 2.379505927685018, 2.3800815376583184, 2.37121504367623, 2.37710815869454,
     2.3745270884823833, 2.3717348159758416, 2.37598854110925, 2.372021871631881, 2.3702655820260388,
     2.3744482782497367, 2.370481156084094, 2.372027556450561, 2.3727048170192258, 2.3722981437499318,
     2.373275267645814, 2.3728743883925163, 2.3722760866985446, 2.373381075849713, 2.3733504629600075, 2.371605797326593, 2.3724627557332187, 2.3717536403166846, 2.372262653717678, 2.3717843510885452, 2.372406481378362]
)

plt.figure(figsize=(15, 8))
plt.tight_layout()
plt.plot(n, Uvych, '-o', markersize=5, label='', color="b")
# plt.legend()
plt.xlabel("Размер сетки")
plt.ylabel("J")
plt.show()

