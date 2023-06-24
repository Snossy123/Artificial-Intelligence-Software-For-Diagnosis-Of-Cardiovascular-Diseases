import wfdb
import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from scipy import signal
import glob

def visualizeSignal(path):
    """
    Load and visualize the ECG signal data from the PhysioNet database.

    Args:
        path (str): The file path of the ECG signal.

    Returns:
        None
    """

    # Load the ECG signal data from the PhysioNet database
    record = wfdb.rdrecord(path, sampto=250)
    signal = record.p_signal[:, 0]

    # Plot the original ECG signal
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title('Original ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

# Load and visualize example ECG signals from different databases
print("Case from MIT-BIH NSRDB")
Normal = visualizeSignal('data/raw/mit-bih-normal-sinus-rhythm-database-1.0.0/mit-bih-normal-sinus-rhythm-database-1.0.0/17453')

print("Case from St INCARTDB DB")
CAD = visualizeSignal('data/raw/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files/I39')

print("Case from PTBDB") 
MI = visualizeSignal('data/raw/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/patient007/s0038lre')

print("Case from BIDMC CHFDB")
CHF = visualizeSignal('data/raw/bidmc-congestive-heart-failure-database-1.0.0/files/chf07')

def checkFiltering(path):
    """
    Load the ECG signal data from the PhysioNet database, 
    calculate its signal-to-noise ratio (SNR), 
    and print the result to the console.

    Args:
        path (str): The file path of the ECG signal.

    Returns:
        snr (float): The signal-to-noise ratio (SNR) of the ECG signal.
    """

    # Load ECG signal 
    record = wfdb.rdrecord(path)
    ecg_signal = record.p_signal[:, 0]

    # Calculate RMS of signal and noise
    signal_rms = np.sqrt(np.mean(ecg_signal**2))
    noise_rms = np.sqrt(np.mean((ecg_signal - signal.medfilt(ecg_signal))**2))

    # Calculate SNR
    snr = 20*np.log10(signal_rms / noise_rms)

    # Print SNR
    print("Signal-to-Noise Ratio (SNR): {:.2f} dB".format(snr))
    
    return snr


# Example usage of checkFiltering function
print("Case from MIT-BIH NSRDB")
print("not need" if checkFiltering('data/raw/mit-bih-normal-sinus-rhythm-database-1.0.0/mit-bih-normal-sinus-rhythm-database-1.0.0/17453')>=10 else "need")

print("Case from St INCARTDB DB")
print("not need" if checkFiltering('data/raw/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files/I39')>=10 else "need")

print("Case from PTBDB")
print("not need" if checkFiltering('data/raw/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/patient007/s0038lre')>=10 else "need")

print("Case from BIDMC CHFDB")
print("not need" if checkFiltering('data/raw/bidmc-congestive-heart-failure-database-1.0.0/files/chf07')>=10 else "need")
 
    
def resampleSignal(path, channels):
    """
    Resample the ECG signal to a consistent sampling rate.

    Parameters:
        path (str): The path to the PhysioNet database record.
        channels (list): The channels to extract from the record.

    Returns:
        resampled_signal (numpy.ndarray): The resampled ECG signal.
    """
    # Load the ECG signal data from the PhysioNet database
    record = wfdb.rdrecord(path, channels=channels)
    signal = record.p_signal[:, 0]

    # Set the original and new sampling rates
    fs = record.fs
    new_fs = 250

    # Create a time array for the original signal
    t_orig = np.arange(signal.size) / fs

    # Create a time array for the resampled signal
    t_new = np.arange(0, t_orig[-1], 1 / new_fs)

    # Resample the ECG signal using linear interpolation
    f = interpolate.interp1d(t_orig, signal)
    resampled_signal = f(t_new)

    return resampled_signal

def pan_tompkins(ecg_signal, fs):
    """
    Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.

    Parameters:
        ecg_signal (numpy.ndarray): The ECG signal.
        fs (float): The sampling rate of the ECG signal.

    Returns:
        r_peaks (numpy.ndarray): An array of indices corresponding to the R-peaks in the ECG signal.
    """
    # Define filter parameters
    lowcut = 5
    highcut = 15
    order = 2

    # Apply band-pass filter to remove unwanted frequencies
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    filtered_ecg_signal = signal.filtfilt(b, a, ecg_signal)

    # Differentiate the signal
    diff_ecg_signal = np.diff(filtered_ecg_signal)

    # Square the differentiated signal
    squared_diff_ecg_signal = diff_ecg_signal ** 2

    # Apply moving average filter to smooth the signal
    window_size = int(0.15 * fs)
    weights = np.ones(window_size) / window_size
    smoothed_squared_diff_ecg_signal = np.convolve(squared_diff_ecg_signal, weights, mode='same')

    # Find the R-peaks
    r_peaks = signal.find_peaks(smoothed_squared_diff_ecg_signal, distance=int(0.2 * fs))[0]

    return r_peaks

# Define function to segment ECG signal into individual heartbeats
def SegmentSignal(peaks, resampled_signal):
    # Set window size for segmenting
    window_size = 200
    heartbeats = []
    # Segment signal into heartbeats centered around detected peaks
    for i in range(len(peaks)):
        start = peaks[i] - int(window_size/2)
        end = peaks[i] + int(window_size/2)
        # Check if the segment falls within the bounds of the signal
        if start < 0 or end > len(resampled_signal):
            continue
        heartbeat = resampled_signal[start:end]
        heartbeats.append(heartbeat)
    # Return segmented heartbeats
    return heartbeats

# Define function to get heartbeats from ECG signal
def getheartbeats(path, channels):
    # Resample ECG signal to a consistent sampling rate
    normal_case = resampleSignal(path, [channels[0]])
    # Detect R-peaks using Pan-Tompkins algorithm
    peaks = pan_tompkins(normal_case, 250)
    # Segment signal into individual heartbeats
    beats1 = SegmentSignal(peaks, normal_case)
    # Repeat for second channel
    normal_case = resampleSignal(path, [channels[1]])
    peaks = pan_tompkins(normal_case, 250)
    beats2 = SegmentSignal(peaks, normal_case)

    # Combine heartbeats from both channels
    heartbeats = [np.concatenate((beat1_elem, beat2_elem), axis=0) for beat1_elem, beat2_elem in zip(beats1, beats2)]
    # Return segmented heartbeats
    return heartbeats

# Define function to generate training data from ECG signals
def generateData(files, channels):
    data = []
    # Loop through files and extract heartbeats from each signal
    for f in normal_files:
        # Get heartbeats for both channels
        heartbeats = getheartbeats(
            f'data/raw/mit-bih-normal-sinus-rhythm-database-1.0.0/mit-bih-normal-sinus-rhythm-database-1.0.0/{f}', [0,1])
        # Append heartbeats to training data
        data += heartbeats
    # Return training data
    return data

# Define lists to hold training and testing data
normal_train = []
normal_test = []
normal_files_train = [['16265'], ['16272'], ['16273'], ['16420'],
                      ['16483'], ['16539'], ['16773'], ['16786'],
                      ['16795'], ['17052']]
normal_files_test = ['19090', '19093', '19140', '19830']

# Load normal sinus rhythm ECG data for training and testing
# Train: Get heartbeats for each file in normal_files_train and append to normal_train
for t in normal_files_train:
    data = []
    for c in t:
        data += getheartbeats(f'data/raw/mit-bih-normal-sinus-rhythm-database-1.0.0/mit-bih-normal-sinus-rhythm-database-1.0.0/{c}', [0,1])
    normal_train.append(data)

# Test: Get heartbeats for each file in normal_files_test and append to normal_test
for ts in normal_files_test:
    normal_test += getheartbeats(f'data/raw/mit-bih-normal-sinus-rhythm-database-1.0.0/mit-bih-normal-sinus-rhythm-database-1.0.0/{ts}', [0,1])

CAD_train = []
CAD_test = []
CAD_files_train = [['I01'], ['I02'], ['I20'], ['I21'],
                   ['I22'], ['I35'], ['I36'], ['I37'],
                   ['I38'], ['I39']]
CAD_files_test = ['I44', 'I45', 'I46', 'I57',
                  'I58', 'I72', 'I73']

# Load CAD ECG data for training and testing
# Train: Get heartbeats for each file in CAD_files_train and append to CAD_train
for t in CAD_files_train:
    data = []
    for c in t:
        data += getheartbeats(f'data/raw/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files/{c}', [1,6])
    CAD_train.append(data)

# Test: Get heartbeats for each file in CAD_files_test and append to CAD_test
for ts in CAD_files_test:
    CAD_test += getheartbeats(f'data/raw/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files/{ts}', [1,6])


# Initialize empty list for MI data
MI = []

# Define folder path and get list of MI files
folder_path = 'data/raw/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0//.dat'
MI_files = glob.glob(folder_path)

# Calculate number of train and test files
train_files_num = int(len(MI_files)*.8/10)*10
test_files_num = len(MI_files) - int(len(MI_files)*.8/10)*10

# Initialize empty lists for MI train and test data
MI_train = []
MI_test = []

# Train files
for i in range(0,train_files_num, int(train_files_num*.1)):
    data = []
    for c in range(i, i+int(train_files_num*.1)):
        data += getheartbeats(MI_files[c][:-4], [1,6])
    MI_train.append(data)

# Test files
for j in range(train_files_num, train_files_num+test_files_num):
    MI_test += getheartbeats(MI_files[j][:-4], [1,6])

# Initialize empty list for CHF data
CHF = []

# Define folder path and get list of CHF files
folder_path = 'data/raw/bidmc-congestive-heart-failure-database-1.0.0/files/*.dat'
CHF_files = glob.glob(folder_path)

# Initialize empty lists for CHF train and test data
CHF_train = []
CHF_test = []

# Train files
for i in range(10):
    CHF_train.append(getheartbeats(CHF_files[i][:-4], [0,1]))

# Test files
for j in range(10, len(CHF_files)):
    CHF_test += getheartbeats(MI_files[j][:-4], [0,1])

# Initialize empty list for minimum train data lengths
min_train = []
for k in range(10):
    min_train.append(min([len(normal_train[k]), len(CAD_train[k]), len(MI_train[k]), len(CHF_train[k])]))

# Function to balance the number of samples for each class
def makeBalance(arr, className, minVal):
    y = [className] * minVal
    df = pd.DataFrame(arr[:minVal])
    df["output"] = y
    return df

# Create balanced datasets for training and testing
for k in range(10):
    # balance the number of samples in each class for the training sets
    normal_train[k] = makeBalance(normal_train[k], 'Normal', min(min_train))
    CAD_train[k] = makeBalance(CAD_train[k], 'CAD', min(min_train))
    MI_train[k] = makeBalance(MI_train[k], 'MI', min(min_train))
    CHF_train[k] = makeBalance(CHF_train[k], 'CHF', min(min_train))

# balance the number of samples in each class for the test set
min_test = min([len(normal_test), len(CAD_test), len(MI_test), len(CHF_test)])
normal_test = makeBalance(normal_test, 'Normal', min_test)
CAD_test = makeBalance(CAD_test, 'CAD', min_test)
MI_test = makeBalance(MI_test, 'MI', min_test)
CHF_test = makeBalance(CHF_test, 'CHF', min_test)

# concatenate the balanced dataframes for each fold and save as a gzipped csv file
dfs_cross_val = []
for k in range(10):
    dfs_cross_val.append(pd.concat([normal_train[k], CAD_train[k], MI_train[k], CHF_train[k]]))
for k in range(10):
    dfs_cross_val[k].to_csv(f'train_data{k}.csv.gz', index=False, compression='gzip')

# concatenate the balanced test set dataframe and save as a gzipped csv file
df_test = pd.concat([normal_test, CAD_test, MI_test, CHF_test])
df_test.to_csv('test_data.csv.gz', index=False, compression='gzip')