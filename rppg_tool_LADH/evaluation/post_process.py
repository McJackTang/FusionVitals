"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_fft_rr(rr_signal, fs=30, low_pass=0.1, high_pass=0.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    rr_signal = np.expand_dims(rr_signal, 0)
    N = _next_power_of_2(rr_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(rr_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    #true_frequency = mask_ppg[np.argmax(mask_pxx)][0]
    # print(f"true_frequency: {true_frequency}")
    fft_rr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _calculate_peak_rr(rr_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(rr_signal)
    rr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return rr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video222(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT', datatype="rppg"):
    """Calculate video-level HR and SNR"""
    if datatype == "rppg":
        print("Calculate rppg_hr", datatype)
        if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
            predictions = _detrend(np.cumsum(predictions), 100)
            labels = _detrend(np.cumsum(labels), 100)
        else:
            predictions = _detrend(predictions, 100)
            labels = _detrend(labels, 100)
        if use_bandpass:
            # bandpass filter between [0.75, 2.5] Hz
            # equals [45, 150] beats per min
            [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
            predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
            labels = scipy.signal.filtfilt(b, a, np.double(labels))
    else:
        print("Calculate rppg_spo2")  
    macc = _compute_macc(predictions, labels)
    print(f"{hr_method}, hr_method, {datatype}, datatype")
    if hr_method == 'FFT' and datatype == "rppg":
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'FFT' and datatype == "spo2":
        hr_pred = predictions
        hr_label = labels
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    if datatype == "rppg":
        SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    else:
        SNR = 0
    return hr_label, hr_pred, SNR, macc
    
def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT', datatype="rppg"):
    """Calculate video-level HR and SNR"""
    # print(f"Processing datatype: {datatype} with HR method: {hr_method}")

    # Detrending and bandpass filtering for rPPG signals
    if datatype == "rppg":
        if diff_flag:
            predictions = _detrend(np.cumsum(predictions), 100)
            labels = _detrend(np.cumsum(labels), 100)
        else:
            predictions = _detrend(predictions, 100)
            labels = _detrend(labels, 100)
        if use_bandpass:
            [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
            predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
            # print(f"predictions: {predictions}")
            labels = scipy.signal.filtfilt(b, a, np.double(labels))
            # print(f"labels: {labels}")

    # HR computation
    hr_pred, hr_label = None, None
    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs) if datatype == "rppg" else predictions
        hr_label = _calculate_fft_hr(labels, fs=fs) if datatype == "rppg" else labels
    elif hr_method == 'Peak' and datatype == "rppg":
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate HR for rPPG.')

    # Calculate SNR only for rPPG data
    SNR = _calculate_SNR(predictions, hr_label, fs=fs) if datatype == "rppg" else None

    # Compute MAcc for all types of data
    macc = _compute_macc(predictions, labels)

    return hr_label, hr_pred, SNR, macc


def calculate_metric_per_video_rr(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT', datatype="rr"):
    """Calculate video-level HR and SNR"""
    print(f"Processing datatype: {datatype} with HR method: {hr_method}")

    # Detrending and bandpass filtering for rPPG signals
    if datatype == "rr":
        if diff_flag:
            predictions = _detrend(np.cumsum(predictions), 100)
            labels = _detrend(np.cumsum(labels), 100)
        else:
            predictions = _detrend(predictions, 100)
            labels = _detrend(labels, 100)
        if use_bandpass:
            [b, a] = butter(1, [0.1 / fs * 2, 0.5 / fs * 2], btype='bandpass')
            predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
            # print(f"predictions: {predictions}")
            labels = scipy.signal.filtfilt(b, a, np.double(labels))
            # print(f"labels: {labels}")
            plot_all(predictions, labels)
            plot_rr_waveforms(predictions, labels)

    # HR computation
    rr_pred, rr_label = None, None
    if hr_method == 'FFT':
        rr_pred = _calculate_fft_rr(predictions, fs=fs) if datatype == "rr" else predictions
        rr_label = _calculate_fft_rr(labels, fs=fs) if datatype == "rr" else labels
    elif hr_method == 'Peak' and datatype == "rr":
        rr_pred = _calculate_peak_rr(predictions, fs=fs)
        rr_label = _calculate_peak_rr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate HR for rPPG.')

    # Calculate SNR only for rPPG data
    SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.1, high_pass=0.5) if datatype == "rr" else None

    # Compute MAcc for all types of data
    macc = _compute_macc(predictions, labels)
    # print(f"rr_label: {rr_label}")
    # print(f"rr_pred: {rr_pred}")
    return rr_label, rr_pred, SNR, macc



def plot_all(rr_pred, rr_lable):
# # plot the filtered RR data and the power spectral density
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    axes[0].plot(rr_pred, label='RR Pred Data')
    axes[0].set_title('RR Data Plot')
    axes[0].set_xlabel('Number')
    axes[0].set_ylabel('Data Value')
    axes[0].legend()

    # # plot the filtered RR data and the power spectral density
    axes[1].plot(rr_lable, label='RR Lable Data')
    axes[1].set_title('RR Data Plot')
    axes[1].set_xlabel('Number')
    axes[1].set_ylabel('Data Value')
    axes[1].legend()

    plt.savefig('./rr_all.png')
    plt.show()

def plot_rr_waveforms(rr_pred, rr_label):

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(rr_pred, label="Predicted RR", color='blue', linewidth=1.5)
    plt.plot(rr_label, label="True RR", color='red', linewidth=1.5)
    
    plt.title("RR ")
    plt.xlabel("number")
    plt.ylabel("RR Rate")
    plt.legend()
    plt.savefig('./tong_rr_lv.png')
    plt.show()