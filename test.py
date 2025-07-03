import numpy as np
import matplotlib.pyplot as plt

pos = np.arange(50)    # 位置 0~49
d_model = 8            # 简化为8维
i = np.arange(d_model // 2)

# 计算波长因子
wavelength = 1 / (10000 ** (2 * i / d_model))

# 生成位置编码
pe = np.zeros((len(pos), d_model))
for p in pos:
    pe[p, 0::2] = np.sin(p * wavelength)  # 偶数维：sin
    pe[p, 1::2] = np.cos(p * wavelength)   # 奇数维：cos

# 绘制热力图
plt.figure(figsize=(10,4))
plt.imshow(pe.T, cmap='viridis', aspect='auto')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.colorbar()
plt.show()