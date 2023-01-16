import numpy as np
import os
from tensorflow import keras
from ncps import wirings
from ncps.tf import LTC
from ncps.tf import CfC
import matplotlib.pyplot as plt
import seaborn as sns

# 输入和输出数据
N = 48 # 时间序列的长度
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N))
    , np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
print(data_x)
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
print(data_y)
print("data_x.shape: ", str(data_x.shape))
print("data_y.shape: ", str(data_y.shape))

sns.set()
plt.figure(figsize=(6, 4))
plt.plot(data_x[0, :, 0], label="Input feature 1")
plt.plot(data_x[0, :, 1], label="Input feature 2")
plt.plot(data_y[0, :, 0], label="Target output")
plt.title("Training data")
plt.legend(loc="upper right")
plt.show()

# 创建一个由 8个全连接神经元组成的LTC/CfC网络，这些神经元接收2个输入特征的时间序列作为输入。此外，我们定义8个神经元中的1个作为输出
fc_wiring = wirings.FullyConnected(8, 1)
model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 2)), # 输入的特征值个数
        # LTC(fc_wiring, return_sequences=True),
        CfC(fc_wiring, return_sequences=True),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error'
)
model.summary()



# 训练前的数据可视化
sns.set()
prediction = model(data_x).numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y[0, :, 0], label="Target output")
plt.plot(prediction[0, :, 0], label="CfC output")
plt.title("Before training")
plt.legend(loc="upper right")
plt.show()
# 训练
hist = model.fit(x=data_x, y=data_y, batch_size=1, epochs=400,verbose=1)
# 训练loss值可视化
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Training loss")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.show()
# 训练后的数据可视化
# 训练加预测数据
# print(np.vstack((data_x[0],data_x[0])))
data_x_pre = np.vstack((data_x[0],data_x[0])) # 两个输入的时间序列
data_x_pre = np.expand_dims(data_x_pre, axis=0).astype(np.float32)
print(data_x_pre)
prediction = model(data_x_pre).numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y[0, :, 0], label="Target output")
plt.plot(prediction[0, :, 0], label="CfC output",linestyle="dashed")
plt.legend(loc="upper right")
plt.title("After training")
plt.show()

