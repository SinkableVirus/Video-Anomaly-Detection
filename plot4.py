import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load data
y = np.load('conv_lstm/conv_lstm.npy')
y2 = np.load('conv_ae/conv_ae.npy')
x = np.array([i for i in range(len(y) + 1)])

# Normalize data
mini = np.min(y)
maxi = np.max(y)
y = (y - mini) / (maxi - mini)

mini2 = np.min(y2)
maxi2 = np.max(y2)
y2 = (y2 - mini2) / (maxi2 - mini2)

# Define the range of frames to plot
low = 330
high = 1700

# Create spline for the first set of data (y)
X_Y_Spline = make_interp_spline(x[low:high], y[low:high])
X_ = np.linspace(x[low:high].min(), x[low:high].max())
Y_ = X_Y_Spline(X_)

# Plot the first set of data
plt.plot(X_, Y_, label='Conv_LSTM', color='blue')

# Create spline for the second set of data (y2)
X_Y_Spline_y2 = make_interp_spline(x[low:high], y2[low:high])
Y2_ = X_Y_Spline_y2(X_)

# Plot the second set of data with a different color
plt.plot(X_, Y2_, label='Conv_AE', color='red')

# Set plot labels and ticks
plt.xlabel("Frame number", fontsize=40)
plt.ylabel("Anomaly score", fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.ylim(0, 0.8)

# Add a legend
plt.legend(fontsize=40)

# Show the plot
plt.show()
