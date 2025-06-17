import numpy as np

# 创建一个示例数组
# pred = np.random((1,1, 18900, 85))
pred = np.random.uniform(-1, 1, (1,1, 7, 7))
print(pred)
print(pred[0][0][:, 4])
print(pred[0][..., 4])
