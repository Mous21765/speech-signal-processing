import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm



# Plot function

def plot(signal):
    plt.specgram(signal, Fs=8000, cmap='inferno', NFFT=256, noverlap=128)

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram of signal')

    # Show the colorbar
    plt.colorbar(format='%+2.0f dB')

    # Display the spectrogram
    plt.show()

# Reading all the recordings from the "files" folder and storing it into the data list
data = []
Y = []
f0s = []
audiofiles = [entry for entry in os.scandir('files') if entry.name.endswith('.wav')]

for record in audiofiles:
    sr, wave = wavfile.read(os.path.join('files', record.name))
    wave = signal.resample(wave,8000)

    Y.append(record.name[0])
    data.append(wave)

    # Calculating autocorrelation
    autocorr = correlate(wave, wave)
    # Finding the peaks in autocorrelation
    peaks, _ = find_peaks(autocorr)

    # If peaks exist then the fundamental frequency is sr/peaks[0]
    if len(peaks) > 0:
        fund_freq = sr / peaks[0]
        f0s.append(fund_freq)


# Printing the fundamental frequency of signal 100,300,400 and 600
print("Fundamental frequency of signal 100: " + str(f0s[100]))

print("Fundamental frequency of signal 300: " + str(f0s[300]))

print("Fundamental frequency of signal 400: " + str(f0s[400]))

print("Fundamental frequency of signal 600: " + str(f0s[600]))


# Adding 0s to the end of all the signals to match the max signal
max_signal = max(map(len, data))
signals = np.vstack([np.pad(x, (0, max_signal - len(x)), mode='constant', constant_values=0) for x in data])

# Getting the spectogram of each signal
specs = np.array([signal.spectrogram(wave, 8000)[2] for wave in signals])


# Plotting spectogram of signal 100 and singal 600

plot(specs[100])
plot(specs[300])
plot(specs[400])
plot(specs[600])

# Scaling the signals
scalers = [MinMaxScaler() for _ in range(specs.shape[1])]
specs = np.array([scalers[i].fit_transform(specs[:, i, :]) for i in range(specs.shape[1])]).transpose(1, 0, 2)

# Reshape to match the SVM
samples = 3000
data = 129 * 35
specs = specs.reshape(samples, data)

# Splitting in training and test batches
X_train, X_test, Y_train, Y_test = train_test_split(specs, Y, train_size=0.7)

# Getting a SVM and fitting the X_TRAIN
lin_clf = svm.LinearSVC(tol=1e-5)
results = lin_clf.fit(X_train, Y_train)

# Predicting using the X_TEST
y_pred = lin_clf.predict(X_test)

print("Digit on signal 100 is " + str(Y_test[100]) + " and SVM predicted " + str(y_pred[100]))
print("Digit on signal 300 is " + str(Y_test[300]) + " and SVM predicted " + str(y_pred[300]))
print("Digit on signal 400 is " + str(Y_test[400]) + " and SVM predicted " + str(y_pred[400]))
print("Digit on signal 600 is " + str(Y_test[600]) + " and SVM predicted " + str(y_pred[600]))

from sklearn import metrics

# Model Accuracy
print("Total accuracy of SVM on X_test:", metrics.accuracy_score(Y_test, y_pred))




