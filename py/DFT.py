"""
    DFT zdefiniowane jest jako

    X_f_k = sum(x * W_kn_N)

    gdzie x(n), n = 0, 1, ... , N - 1 to proba sygnalu probkowanego rownomierne tzn.
    co dokladnie odpowiedni interwal (tzw. "sampling interval"), N to liczba prob ("sampling rate")

    W_N = exp(-j2PI/N)

    jest N-tym pierwiastkiem z jednosci,

    W_kn_N = exp((-j2PI/N)kn) oraz

    X_f_k, k = 0,1, ... , N - 1

    jest k-tym wspolczynnikiem DFT,

    j = math.sqrt(-1)
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import math

"""
    Funkcja generuje sygnal, zwraca jego "samplingRate" prob 
"""
def generateSignal(samplingRate):
    """
        Sygnal w domenie czasu gdzie 0 <= t <= 1,
        probkujemy dokladnie "samplingRate" punktow co "samplingInterval"
    """

    samplingInterval = 1.0 / samplingRate
    t = np.arange(0, 1, samplingInterval)

    """
        'Hardcodujemy' sygnal "x" poprzez sume pewnych funkcji sinus,
        odpowiednio je skladamy oraz skalujemy aby mialy odpowiednia 
        czestotliwosc oraz amplitude, gdzie
        
        frequency - czestotliwosc sygnalu
        amplitude - amplituda sygnalu
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
        Funkcja proby sygnalu x : t -> R wyraz sie jako
        
        x(t) = x_1(t) + x_2(t) + x_3(t)
    """

    return [t, x, samplingInterval]

"""
    Funkcja zwraca wspolczynnik W_kn_N     
"""

def getW_kn_N(k, n, N):
    return np.exp(((-1j * 2 * np.pi) / N) * k * n)

"""
    Funkcja liczy DFT sygnalu "x" oraz zwraca k-ty wspolczynnik Fouriera
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
signal = generateSignal(samplingRate)

t = signal[0]
x = signal[1]
samplingInterval = signal[2]
N = len(t)

plt.figure(figsize=(9, 7))
plt.plot(t, x, 'r', marker='o', linestyle=':', markersize=4)
plt.xlabel('Czas (s)')
plt.ylabel('Probka')
plt.title('Wykres proby sygnalu')
plt.show()

X_f = np.array([complex(0, 0)] * N, dtype = complex)

sTime = time.time()

for k in range(N):
    X_f[k] = getDFT(k, x)

eTime = time.time()
timeElapsed = eTime - sTime
print("Czas liczenia %d wspolczynnikow DFT = %0.5es" % (N, timeElapsed))

n = np.arange(N)
T = N / samplingRate
frequencyDomain = n / T

plt.figure(figsize=(9, 7))
plt.stem(frequencyDomain, abs(X_f), 'b', markerfmt="bo", basefmt="-b")
plt.xlabel('Czestotliwosc (Hz)')
plt.ylabel('|X_f(frequency)|')
plt.title('Spektrum magnitudy Fouriera')
plt.show()
