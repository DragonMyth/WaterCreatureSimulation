import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 2 * np.pi, 0.1)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
plt.plot(x, y, color='pink', linewidth='15')
plt.text(-6, 2, "Xiao Miao Miao", color='pink', fontsize='xx-large')
plt.text(-4, -2, "Love You~~", color='pink', fontsize='xx-large')
plt.axis('off')
plt.show()
