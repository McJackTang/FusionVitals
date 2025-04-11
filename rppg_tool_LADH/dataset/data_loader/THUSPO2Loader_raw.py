import numpy as np
import pandas as pd
import cv2
import glob
import os
from scipy.interpolate import interp1d
from dataset.data_loader.BaseLoader import BaseLoader


class THUSPO2Loader(BaseLoader):
    def __init__(self, name, data_path, config_data):
        """Initializes an THUSPO2 dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 070200/v01
                     |       |-- video_ZIP_H264_face.avi
                     |       |-- BVP.csv
                     |       |-- frames_timestamp.csv
                     |       |-- SpO2.csv
                     |       |-- RR.csv(some may have RR)
                     |   |-- 070201/v01
                     |       |-- video_ZIP_H264_face.avi
                     |       |-- BVP.csv
                     |       |-- frames_timestamp.csv
                     |       |-- SpO2.csv
                     |       |-- RR.csv(some may have RR)
                     |...
                     |   |-- 080305/v01
                     |       |-- video_ZIP_H264_face.avi
                     |       |-- BVP.csv
                     |       |-- frames_timestamp.csv
                     |       |-- SpO2.csv
                     |       |-- RR.csv(some may have RR)
                     |   |-- 080305/v02
                     |       |-- video_ZIP_H264_face.avi
                     |       |-- BVP.csv
                     |       |-- frames_timestamp.csv
                     |       |-- SpO2.csv
                     |       |-- RR.csv(some may have RR)
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For THUSPO2 dataset)."""
        # data_dirs = glob.glob(os.path.join(data_path, "*", "v*", "video_ZIP_H264_face.avi"))
        data_dirs = glob.glob(os.path.join(data_path, "*", "v*", "video_ZIP_H264_finger.avi"))
        # 调用了重写的get_raw_data方法
        # print('glob data dirslajsfkldjfkfjdslkjfkl调用了子类', data_dirs)
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": os.path.basename(data_dir).split('.')[0], "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess_raw(self, data_dirs, config_preprocess, i, file_list_dict):
        # Read video frames
        
        video_file = data_dirs[i]['path']
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)

        # Extract subject ID from the directory path
        # Adjust index according to your directory structure
        # print(video_dir.split(os.sep))
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID

        # Get BVP and frame timestamp files
        bvp_file = os.path.join(video_dir, "BVP.csv")
        timestamp_file = os.path.join(video_dir, "frames_timestamp.csv")

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Resample BVP to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)

        # Process frames and BVP signals according to the configuration
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = resampled_bvp

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        # Create a unique filename that includes the subject ID and experiment ID
        filename = f"{subject_id}_{experiment_id}_{data_dirs[i]['index']}"

        # Save processed data and update the file list dictionary with filenames including subject ID and experiment ID
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, filename)
        file_list_dict[i] = input_name_list

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        # Read video frames
        print("执行了preprocess_dataset_subprocess")
        video_file = data_dirs[i]['path']
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)

        # Extract subject ID from the directory path
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID

        # Get BVP, frame timestamp, and SpO2 files
        bvp_file = os.path.join(video_dir, "BVP.csv")
        timestamp_file = os.path.join(video_dir, "frames_timestamp.csv")
        spo2_file = os.path.join(video_dir, "SpO2.csv")

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Read SpO2 data and timestamps
        spo2_timestamps, spo2_values = self.read_spo2(spo2_file)

        # Resample BVP to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)

        # Resample SpO2 to match video frames
        resampled_spo2 = self.synchronize_and_resample(spo2_timestamps, spo2_values, frame_timestamps)

        # Process frames, BVP signals, and SpO2 signals according to the configuration
        
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            # 走的这个
            bvps = resampled_bvp
            spo2 = resampled_spo2

        frames_clips, bvps_clips, spo2_clips = self.preprocess(frames, bvps, spo2, config_preprocess)

        # Create a unique filename that includes the subject ID and experiment ID
        filename = f"{subject_id}_{experiment_id}_{data_dirs[i]['index']}"

        # Save processed data and update the file list dictionary with filenames including subject ID and experiment ID
        input_name_list, label_name_list, spo2_name_list = self.save_multi_process(frames_clips, bvps_clips, spo2_clips,filename)
        # spo2_name_list = self.save_spo2(spo2_clips, filename)  # Save SpO2 data separately if needed

        file_list_dict[i] = input_name_list

    @staticmethod
    def read_spo2(spo2_file):
        """Reads a SpO2 signal file with timestamps."""
        data = pd.read_csv(spo2_file)
        timestamps = data['timestamp'].values
        spo2_values = data['spo2'].values
        return timestamps, spo2_values

    @staticmethod
    def read_bvp(bvp_file):
        """Reads a BVP signal file with timestamps."""
        data = pd.read_csv(bvp_file)
        timestamps = data['timestamp'].values
        bvp_values = data['bvp'].values
        return timestamps, bvp_values

    @staticmethod
    def read_frame_timestamps(timestamp_file):
        """Reads timestamps for each video frame."""
        data = pd.read_csv(timestamp_file)
        return data['timestamp'].values

    @staticmethod
    def synchronize_and_resample(timestamps_data, data_values, timestamps_frames):
        """Synchronize and resample data to match video frame timestamps."""
        interpolator = interp1d(timestamps_data, data_values, bounds_error=False, fill_value="extrapolate")
        resampled_data = interpolator(timestamps_frames)
        return resampled_data


    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.array(frames)

    
    def get_dataset_files(self, data_path):
        """Retrieve the first set of files for a dataset directory."""
        # Find the first directory with video and CSV files
        video_file = glob.glob(os.path.join(data_path, "video_ZIP_H264_face.avi"))[0]
        bvp_file = glob.glob(os.path.join(data_path, "BVP.csv"))[0]
        timestamp_file = glob.glob(os.path.join(data_path, "frames_timestamp.csv"))[0]
        spo2_file = glob.glob(os.path.join(data_path, "SpO2.csv"))[0]

        return video_file, bvp_file, timestamp_file, spo2_file


    def test_bvp_resampling(self, dataset_path):
            """Test function to check BVP data synchronization and resampling using the first available file."""
            # Retrieve the first dataset files
            video_file, bvp_file, timestamp_file, spo2_file = self.get_dataset_files(dataset_path)

            # Read video frames and timestamps
            frames = self.read_video(video_file)
            frame_timestamps = self.read_frame_timestamps(timestamp_file)

            # Read BVP data and timestamps
            bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

            # Read SpO2 data and timestamps
            spo2_timestamps, spo2_values = self.read_spo2(spo2_file)

            # Resample BVP to match video frames
            resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)

            # Resample SpO2 to match video frames
            resampled_spo2 = self.synchronize_and_resample(spo2_timestamps, spo2_values, frame_timestamps)

            print("Number of frames: ", len(frames))
            print("Number of resampled BVP data points: ", len(resampled_bvp))
            print("Number of resampled SpO2 data points: ", len(resampled_spo2))

            return resampled_bvp, resampled_spo2

