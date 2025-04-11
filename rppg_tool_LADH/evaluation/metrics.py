import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from sklearn.decomposition import PCA
from evaluation.BlandAltmanPy import BlandAltman
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics3333(predictions, labels, config, datatype):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print(f"{datatype} ----------------- Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT', datatype=datatype)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x

    gt_hr_fft_all = np.array([to_numpy(tensor) for tensor in gt_hr_fft_all])
    predict_hr_fft_all = np.array([to_numpy(tensor) for tensor in predict_hr_fft_all])
    if datatype == "rppg":
        SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
    MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    metrics_result = {}

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        num_test_samples = len(predict_hr_fft_all)
        
        # Convert predict_hr_fft_al and gt_hr_fft_all to 1D arrays where elements have unequal lengths
        predict_hr_fft_all = [np.array(x, ndmin=1) for x in predict_hr_fft_all]
        predict_hr_fft_all = np.concatenate(predict_hr_fft_all)
        gt_hr_fft_all = [np.array(x, ndmin=1) for x in gt_hr_fft_all]
        gt_hr_fft_all = np.concatenate(gt_hr_fft_all)
    
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                MAE_STD = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_MAE"] = (MAE_FFT, MAE_STD)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_STD))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                RMSE_STD = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_RMSE"] = (RMSE_FFT, RMSE_STD)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_STD))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["FFT_MAPE"] = (MAPE_FFT, MAPE_STD)
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_STD))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["FFT_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_FFT = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["FFT_SNR"] = (SNR_FFT, SNR_STD)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, SNR_STD))
                else:
                    print("FFT SNR (FFT Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                # make sure the data passed to BlandAltman is one-dimensional
                gt_hr_fft_all_flat = gt_hr_fft_all.flatten()
                predict_hr_fft_all_flat = predict_hr_fft_all.flatten()
                try:
                    compare = BlandAltman(gt_hr_fft_all_flat, predict_hr_fft_all_flat, config, averaged=True)
                    if datatype == "rppg":
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_HR',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_HR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_HR',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_HR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2.pdf',
                            measure_lower_lim=85, measure_upper_lim=100)
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2.pdf')
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array([to_numpy(tensor) for tensor in gt_hr_peak_all])
        predict_hr_peak_all = np.array([to_numpy(tensor) for tensor in predict_hr_peak_all])
        if datatype == "rppg":
            SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
        MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])
        
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                MAE_STD = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_MAE"] = (MAE_PEAK, MAE_STD)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, MAE_STD))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                RMSE_STD = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_RMSE"] = (RMSE_PEAK, RMSE_STD)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, RMSE_STD))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["PEAK_MAPE"] = (MAPE_PEAK, MAPE_STD)
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, MAPE_STD))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["PEAK_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_PEAK = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["PEAK_SNR"] = (SNR_PEAK, SNR_STD)
                    print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_PEAK, SNR_STD))
                else:
                    print("PEAK SNR (Peak Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                gt_hr_peak_all_flat = gt_hr_peak_all.flatten()
                predict_hr_peak_all_flat = predict_hr_peak_all.flatten()
                try:
                    compare = BlandAltman(gt_hr_peak_all_flat, predict_hr_peak_all_flat, config, averaged=True)
                    if datatype == "rppg":
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_HR',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_HR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_HR',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_HR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2.pdf')
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2.pdf') 
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    if datatype == "rppg":
        return metrics_result["FFT_MAE"], metrics_result["FFT_RMSE"], metrics_result["FFT_MAPE"], metrics_result["FFT_Pearson"], metrics_result["FFT_SNR"]
    else:
        return metrics_result["FFT_MAE"], metrics_result["FFT_RMSE"], metrics_result["FFT_MAPE"], metrics_result["FFT_Pearson"]

def calculate_metrics(predictions, labels, config, datatype):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print(f"{datatype} ----------------- Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        #print(f"video_frame_size :{video_frame_size}")
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        #print(f"window_frame_size :{window_frame_size}")
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            #print(f"pred_window :{pred_window}")
            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT', datatype=datatype)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x

    # plot_rr_waveforms(predict_hr_fft_all, gt_hr_fft_all)
    # gt_hr_fft_all = np.array([to_numpy(tensor) for tensor in gt_hr_fft_all], dtype=object)
    # predict_hr_fft_all = np.array([to_numpy(tensor) for tensor in predict_hr_fft_all], dtype=object)
    gt_hr_fft_all = np.array([to_numpy(tensor) for tensor in gt_hr_fft_all])
    predict_hr_fft_all = np.array([to_numpy(tensor) for tensor in predict_hr_fft_all])
    if datatype == "rppg":
        SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
    MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    metrics_result = {}

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        num_test_samples = len(predict_hr_fft_all)
        
        predict_hr_fft_all = [np.array(x, ndmin=1) for x in predict_hr_fft_all]
        predict_hr_fft_all = np.concatenate(predict_hr_fft_all)
        gt_hr_fft_all = [np.array(x, ndmin=1) for x in gt_hr_fft_all]
        gt_hr_fft_all = np.concatenate(gt_hr_fft_all)
    
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                MAE_STD = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_MAE"] = (MAE_FFT, MAE_STD)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_STD))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                RMSE_STD = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_RMSE"] = (RMSE_FFT, RMSE_STD)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_STD))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["FFT_MAPE"] = (MAPE_FFT, MAPE_STD)
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_STD))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["FFT_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_FFT = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["FFT_SNR"] = (SNR_FFT, SNR_STD)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, SNR_STD))
                else:
                    print("FFT SNR (FFT Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:

                gt_hr_fft_all_flat = gt_hr_fft_all.flatten()
                predict_hr_fft_all_flat = predict_hr_fft_all.flatten()
                try:
                    compare = BlandAltman(gt_hr_fft_all_flat, predict_hr_fft_all_flat, config, averaged=True)
                    if datatype == "rppg":
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_HR',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_HR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_HR',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_HR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2.pdf',
                            measure_lower_lim=85, measure_upper_lim=100)
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2.pdf')
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array([to_numpy(tensor) for tensor in gt_hr_peak_all])
        predict_hr_peak_all = np.array([to_numpy(tensor) for tensor in predict_hr_peak_all])
        if datatype == "rppg":
            SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
        MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])
        
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                MAE_STD = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_MAE"] = (MAE_PEAK, MAE_STD)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, MAE_STD))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                RMSE_STD = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_RMSE"] = (RMSE_PEAK, RMSE_STD)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, RMSE_STD))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["PEAK_MAPE"] = (MAPE_PEAK, MAPE_STD)
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, MAPE_STD))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["PEAK_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_PEAK = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["PEAK_SNR"] = (SNR_PEAK, SNR_STD)
                    print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_PEAK, SNR_STD))
                else:
                    print("PEAK SNR (Peak Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                gt_hr_peak_all_flat = gt_hr_peak_all.flatten()
                predict_hr_peak_all_flat = predict_hr_peak_all.flatten()
                try:
                    compare = BlandAltman(gt_hr_peak_all_flat, predict_hr_peak_all_flat, config, averaged=True)
                    if datatype == "rppg":
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_HR',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_HR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_HR',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_HR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2.pdf')
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2.pdf') 
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    # Create the result dictionary to return
    result = {
        "metrics": metrics_result,
        "datatype": datatype
    }

    return result

def calculate_metrics_RR(predictions, labels, config, datatype):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_rr_fft_all = list()
    gt_rr_fft_all = list()
    predict_rr_peak_all = list()
    gt_rr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print(f"{datatype} ----------------- Calculating metrics!")
    # print(f"predictions: {predictions}")
    # print(predictions.shape)
    # print(f"labels: {labels}")
    # print(labels.shape)
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])
        # print(f"prediction: {prediction}")
        # print(f"label: {label}")

        video_frame_size = prediction.shape[0]
        # print(f"video_frame_size: {video_frame_size}")
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        # print(f"window_frame_size: {window_frame_size}")
        # print("len(prediction):")
        # print(len(prediction))
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            # print("len(pred_window):")
            # print(len(pred_window))
            # print("len(label_window):")
            # print(len(label_window))
            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_rr_peak, pred_rr_peak, SNR, macc = calculate_metric_per_video_rr(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_rr_peak_all.append(gt_rr_peak)
                predict_rr_peak_all.append(pred_rr_peak)
                if datatype == "rr":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                # print(f"pred_window: {pred_window}")
                # print(f"label_window: {label_window}")
                plot_rr_window(pred_window, label_window)
                plot_rr_waveforms(pred_window, label_window)
                gt_rr_fft, pred_rr_fft, SNR, macc = calculate_metric_per_video_rr(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                # print(f"gt_rr_fft: {gt_rr_fft}")
                gt_rr_fft_all.append(gt_rr_fft)
                predict_rr_fft_all.append(pred_rr_fft)
                if datatype == "rr":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x
    # plot_pred(predict_rr_fft_all)
    # plot_lable(gt_rr_fft_all)
    # print(f"gt_rr_fft_all: {gt_rr_fft_all}")
    # print(f"predict_rr_fft_all: {predict_rr_fft_all}")
    plot_rr_waveforms(predict_rr_fft_all, gt_rr_fft_all)
    gt_rr_fft_all = np.array([to_numpy(tensor) for tensor in gt_rr_fft_all])
    predict_rr_fft_all = np.array([to_numpy(tensor) for tensor in predict_rr_fft_all])
    if datatype == "rr":
        SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
    MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    metrics_result = {}

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        num_test_samples = len(predict_rr_fft_all)
        
        predict_rr_fft_all = [np.array(x, ndmin=1) for x in predict_rr_fft_all]
        predict_rr_fft_all = np.concatenate(predict_rr_fft_all)
        gt_rr_fft_all = [np.array(x, ndmin=1) for x in gt_rr_fft_all]
        gt_rr_fft_all = np.concatenate(gt_rr_fft_all)
    
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_rr_fft_all - gt_rr_fft_all))
                MAE_STD = np.std(np.abs(predict_rr_fft_all - gt_rr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_MAE"] = (MAE_FFT, MAE_STD)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_STD))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_rr_fft_all - gt_rr_fft_all)))
                RMSE_STD = np.std(np.square(predict_rr_fft_all - gt_rr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_RMSE"] = (RMSE_FFT, RMSE_STD)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_STD))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) * 100
                MAPE_STD = np.std(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["FFT_MAPE"] = (MAPE_FFT, MAPE_STD)
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_STD))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_rr_fft_all, gt_rr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["FFT_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rr":
                if len(SNR_all) > 0:
                    SNR_FFT = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["FFT_SNR"] = (SNR_FFT, SNR_STD)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, SNR_STD))
                else:
                    print("FFT SNR (FFT Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                gt_rr_fft_all_flat = gt_rr_fft_all.flatten()
                predict_rr_fft_all_flat = predict_rr_fft_all.flatten()
                # print(f"test_batgt_rr_fft_all_flat: {gt_rr_fft_all_flat}")
                # print(f"predict_rr_fft_all_flat: {predict_rr_fft_all_flat}")
                try:
                    compare = BlandAltman(gt_rr_fft_all_flat, predict_rr_fft_all_flat, config, averaged=True)
                    if datatype == "rr":
                        compare.scatter_plot(
                            x_label='GT PPG RR [bpm]',
                            y_label='rPPG RR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_RR',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_RR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG RR and GT PPG RR [bpm]',
                            y_label='Average of rPPG RR and GT PPG RR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_RR',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_RR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot_SPO2.pdf',
                            measure_lower_lim=85, measure_upper_lim=100)
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot_SPO2.pdf')
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_rr_peak_all = np.array([to_numpy(tensor) for tensor in gt_rr_peak_all])
        predict_rr_peak_all = np.array([to_numpy(tensor) for tensor in predict_rr_peak_all])
        if datatype == "rr":
            SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
        MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])
        
        num_test_samples = len(predict_rr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_rr_peak_all - gt_rr_peak_all))
                MAE_STD = np.std(np.abs(predict_rr_peak_all - gt_rr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_MAE"] = (MAE_PEAK, MAE_STD)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, MAE_STD))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_rr_peak_all - gt_rr_peak_all)))
                RMSE_STD = np.std(np.square(predict_rr_peak_all - gt_rr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_RMSE"] = (RMSE_PEAK, RMSE_STD)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, RMSE_STD))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) * 100
                MAPE_STD = np.std(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["PEAK_MAPE"] = (MAPE_PEAK, MAPE_STD)
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, MAPE_STD))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_rr_peak_all, gt_rr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["PEAK_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rr":
                if len(SNR_all) > 0:
                    SNR_PEAK = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["PEAK_SNR"] = (SNR_PEAK, SNR_STD)
                    print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_PEAK, SNR_STD))
                else:
                    print("PEAK SNR (Peak Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                gt_rr_peak_all_flat = gt_rr_peak_all.flatten()
                predict_rr_peak_all_flat = predict_rr_peak_all.flatten()
                try:
                    compare = BlandAltman(gt_rr_peak_all_flat, predict_rr_peak_all_flat, config, averaged=True)
                    if datatype == "rr":
                        compare.scatter_plot(
                            x_label='GT PPG RR [bpm]',
                            y_label='rPPG RR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_RR',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_RR.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG RR and GT PPG RR [bpm]',
                            y_label='Average of rPPG RR and GT PPG RR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_RR',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_RR.pdf')
                    elif datatype == "spo2":
                        compare.scatter_plot(
                            x_label='GT SPO2 [%]',
                            y_label='Predicted SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot_SPO2.pdf')
                        compare.difference_plot(
                            x_label='Difference between Predicted SPO2 and GT SPO2 [%]',
                            y_label='Average of Predicted SPO2 and GT SPO2 [%]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot_SPO2.pdf') 
                except np.linalg.LinAlgError:
                    print("The data covariance matrix is ​​singular and a Gaussian kernel density estimate cannot be computed.")
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    # Create the result dictionary to return
    result = {
        "metrics": metrics_result,
        "datatype": datatype
    }

    return result

# Example usage:
# result = calculate_metrics(predictions, labels, config, "rppg")
# print(result["metrics"]["FFT_MAE"])



def plot_rr_waveforms(rr_pred, rr_label):
    """
    Plot waveforms of respiratory rate predictions and labels.
    Args:
        rr_pred: Respiratory rate predictions (tensor or list-like).
        rr_label: Respiratory rate labels (tensor or list-like).
        fs: Sampling frequency (default=30 Hz).
    """
    # Ensure the data is converted to numpy arrays
    rr_pred = np.array(rr_pred)
    rr_label = np.array(rr_label)

    # Create a time axis based on the length of the signals
    time_axis = np.arange(len(rr_pred))  # Time in seconds

    # Plotting
    plt.figure(figsize=(20, 15))
    plt.plot(time_axis, rr_pred, label="Predicted RR", color='blue', linewidth=1.5)
    plt.plot(time_axis, rr_label, label="True RR", color='red', linewidth=1.5)
    
    plt.title("RR Rate")
    plt.xlabel("number")
    plt.ylabel("RR Rate (breaths per minute)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig('./result.png')
    plt.show()

def plot_rr_window(rr_pred, rr_lable):

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

    plt.savefig('./rr_window.png')
    plt.show()

def plot_rr(rr_pred, rr_lable):

    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    axes[0].plot(rr_pred, label='RR Pred')
    axes[0].set_title('RR Data Plot')
    axes[0].set_xlabel('Number')
    axes[0].set_ylabel('Data Value')
    axes[0].legend()

    # # plot the filtered RR data and the power spectral density
    axes[1].plot(rr_lable, label='RR Lable')
    axes[1].set_title('RR Data Plot')
    axes[1].set_xlabel('Number')
    axes[1].set_ylabel('Data Value')
    axes[1].legend()

    plt.savefig('./lable_pred.png')
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
    plt.savefig('./tong_rr.png')
    plt.show()