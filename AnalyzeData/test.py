import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time EMG (Fast)")
plot = win.addPlot(title="EMG Signal")
curve = plot.plot(pen='y')
win.show()

data = np.zeros(500)

def update():
    global data
    new_val = np.random.normal()
    data = np.roll(data, -1)
    data[-1] = new_val
    curve.setData(data)

# 设置 QTimer：每 1ms 调用一次 update（=1000fps理论值）
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

# 启动 GUI 事件循环
sys.exit(app.exec_())
