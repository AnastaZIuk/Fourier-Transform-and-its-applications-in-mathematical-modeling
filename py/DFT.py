import matplotlib.pyplot as plt
import numpy as np
import time

"""
    Generates a signal, returns it's samples - amount of samples is given by "samplingRate"
"""

def generateSignal(samplingRate, samplingInterval = 1.0):
    """
        Signal function "x" is in time domain where we hardcode that 0 <= t <= "samplingInterval",
        we do sample "samplingRate" points distributed uniformly with "uniformStep"
    """

    uniformStep = samplingInterval / samplingRate
    t = np.arange(0, samplingInterval, uniformStep)

    """
        We do generate a signal "x" as a sum of certain composition of sine functions
        scalled by a factor to make sure they have an amplitude and frequency known
        in advance
    """

    frequency = 1.
    amplitude = 3.
    x = amplitude * np.sin(2 * np.pi * frequency * t)  # x_1(t)

    frequency = 4.
    amplitude = 1.
    x += amplitude * np.sin(2 * np.pi * frequency * t)  # x_2(t)

    frequency = 10.
    amplitude = 0.5
    x += amplitude * np.sin(2 * np.pi * frequency * t)  # x_3(t)

    """
        Final signal x : t -> R is

        x(t) = x_1(t) + x_2(t) + x_3(t)
    """

    return [t, x, samplingInterval]

"""
    Returns W_kn_N coefficient  
"""

def getW_kn_N(k, n, N):
    return np.exp(((-1j * 2 * np.pi) / N) * k * n)

"""
    Calculates DFT of a signal "x" and return the k-th DFT coefficient
"""

def getDFT(k, x, normalize = True):
    N = len(x)
    n = np.arange(N)
    W_kn_N = getW_kn_N(k, n, N)

    X_f_k = sum(x * W_kn_N)

    if(normalize):
        X_f_k /= N

    return X_f_k

samplingRate = 100
samplingInterval = 1.0

signal = generateSignal(samplingRate, samplingInterval)

t = signal[0]
x = signal[1]
N = len(t)

plt.figure(figsize=(9, 7))
plt.plot(t, x, 'r', marker='o', linestyle=':', markersize=4)
plt.xlabel('Time (s)')
plt.ylabel('Sample value')
plt.title('Samples plot')
plt.show()

X_f = np.array([complex(0, 0)] * N, dtype = complex)

sTime = time.time()

for k in range(N):
    X_f[k] = getDFT(k, x)

eTime = time.time()
timeElapsed = eTime - sTime
print("Time of calculating %d DFT coefficients = %0.5es" % (N, timeElapsed))

frequencyDomain = np.arange(N) / samplingInterval

plt.figure(figsize=(9, 7))
plt.stem(frequencyDomain, abs(X_f), 'b', markerfmt="bo", basefmt="-b")
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X_f(frequency)|')
plt.title('Magnitude Fourier Spectrum')
plt.show()