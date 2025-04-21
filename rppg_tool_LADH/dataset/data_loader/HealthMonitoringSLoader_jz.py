import numpy as np
import pandas as pd
import cv2
import glob
import os
from scipy.interpolate import interp1d
from dataset.data_loader.BaseLoader import BaseLoader
import matplotlib.pyplot as plt

class HealthMonitoringSLoader_jz(BaseLoader):
    def __init__(self, name, data_path, config_data):
        """Initializes an THUSPO2 dataloader."""
        self.info = config_data.INFO  
        print(data_path)
        print("fsahfsjadjfk")
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories in the specified path."""
        print(data_path)
        # print("jz_dataloader")

        # 查找所有以 'mirror_' 开头的文件夹
        data_dirs = glob.glob(data_path + os.sep + 'mirror_*')
        # print("data_dirs = ",data_dirs)
        
        # 如果没有找到文件夹，抛出错误
        if not data_dirs:
            raise ValueError(self.dataset_name + ' Data path is empty!')
        
        dirs = list()

        # 遍历每个镜像文件夹
        for data_dir in data_dirs:
            # 获取镜像文件夹名（例如 mirror_1, mirror_2）
            subject = os.path.split(data_dir)[-1]
            
            # 遍历 v01 到 v04 文件夹
            for v_dir in ['01', '02', '03', '04']:
                v_path = os.path.join(data_dir, v_dir)  # 拼接 v01, v02, v03, v04 文件夹的路径
                # print("v_path=",v_path)
                if not os.path.exists(v_path):
                    continue  # 如果 vXX 文件夹不存在，跳过

                # 检查 Camera1 和 Camera2 文件夹中的 video.avi 文件
                for camera in ['Camera1', 'Camera2']:
                    camera_path = os.path.join(v_path, camera)
                    # print("camera_path=",camera_path)
                    if not os.path.exists(camera_path):
                        print(f"Warning: {camera_path} does not exist.")
                        continue  # 如果 Camera1 或 Camera2 文件夹不存在，跳过
                    if os.path.exists(camera_path):
                        items = os.listdir(camera_path)  # 获取 Camera1 或 Camera2 中的所有文件
                        for item in items:
                            if item == "video.avi":  # 找到 video.avi 文件
                                dirs.append({
                                    'index': v_dir[1:],  # 去掉 'v' 字符
                                    'path': os.path.join(camera_path, item),  # 拼接完整的文件路径
                                    'subject': subject,  # 镜像文件夹名作为 'subject'
                                    'type': camera  # 文件所在的相机类型（Camera1 或 Camera2）
                                })
                    
        return dirs
        
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        print("start split_raw_data")
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        # Split according to tags v01, v02, v03, v04
        
        data_info = dict()
        for data in data_dirs:
            # index = data['index']
            # data_dir = data['path']
            subject = data['subject']
            # type = data['type'] # face or finger
            # Create a data directory dictionary indexed by subject number
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = list(data_info.keys())  # Get all subject numbers
        subj_list = sorted(subj_list)  # Sort subject numbers
        
        num_subjs = len(subj_list)  # Total number of subjects      
        
        # Get data set split (according to start/end ratio)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('Subjects ID used for split:', [subj_list[i] for i in subj_range])

        # Add file paths that meet the split range to the new list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
        
        print(data_dirs_new)
        return data_dirs_new        
        
        


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,  file_list_dict):
        # Read video frames
        video_file = data_dirs[i]['path']
        print("video_file=",video_file)
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)
        print("video_dir=",video_dir)

        # # Extract subject ID and experiment ID from the directory path
        # subject_id = video_dir.split(os.sep)[-2]
        # experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID
        # print(f"subject_id: {subject_id}, experiment_id: {experiment_id}")
        # # Get BVP, frame timestamps, and SpO2 files
        # bvp_file = os.path.join(video_dir, "BVP.csv")
        # timestamp_file = os.path.join(video_dir, "frames_timestamp_RGB.csv")
        # spo2_file = os.path.join(video_dir, "SpO2.csv")
		# # Read RR data and timestamps
        # rr_file = os.path.join(video_dir, "RR.csv")
        # Extract subject ID and experiment ID from the directory path
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID
        name_id = video_dir.split(os.sep)[-3]
        print(f"subject_id: {subject_id}, experiment_id: {experiment_id}")

        # Get the directory of the parent folder (the level above video_dir)
        parent_dir = os.path.dirname(video_dir)

        # Get the full paths of the files based on the new location
        bvp_file = os.path.join(parent_dir, "Oximeter", "bvp.csv")  # bvp.csv is in the Oximeter folder at the same level as video_dir
        timestamp_file = os.path.join(video_dir, "timestamps.csv")  # timestamps.csv is directly in video_dir
        spo2_file = os.path.join(parent_dir, "Oximeter", "spo2.csv")  # spo2.csv is in the Oximeter folder at the same level as video_dir
        rr_file = os.path.join(parent_dir, "Respiration", "resp.csv")  # resp.csv is in the Respiration folder at the same level as video_dir



# Now you can read the files as needed
# Example:
# bvp_data = self.read_bvp(bvp_file)
# timestamp_data = self.read_frame_timestamps(timestamp_file)
# spo2_data = self.read_spo2(spo2_file)
# rr_data = self.read_rr(rr_file)

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Read SpO2 data and timestamps
        spo2_timestamps, spo2_values = self.read_spo2(spo2_file)
		
		# Read RR data and timestamps
        rr_timestamps, rr_values = self.read_rr(rr_file)
        print("finish read")
        # # Calculate and print sampling frequency for BVP, SpO2, RR
        # bvp_sampling_rate = 1 / (bvp_timestamps[1] - bvp_timestamps[0])  # Assuming uniform sampling
        # print(f"BVP sampling frequency: {bvp_sampling_rate} Hz")
        # spo2_sampling_rate = 1 / (spo2_timestamps[1] - spo2_timestamps[0])  # Assuming uniform sampling
        # print(f"SpO2 sampling frequency: {spo2_sampling_rate} Hz")
        # rr_sampling_rate = 1 / (rr_timestamps[1] - rr_timestamps[0])  # Assuming uniform sampling
        # print(f"RR sampling frequency: {rr_sampling_rate} Hz")

        # Resample BVP data to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)
        

        # Resample SpO2 data to match video frames
        resampled_spo2 = self.synchronize_and_resample(spo2_timestamps, spo2_values, frame_timestamps)
        
		# Resample RR data to match video frames
        resampled_rr = self.synchronize_and_resample(rr_timestamps, rr_values, frame_timestamps)
        # plot_all(rr_values, resampled_rr)
        # plot_rr(rr_values, resampled_rr, rr_timestamps, frame_timestamps)
        # # Calculate the time difference between consecutive frame timestamps
        # resampled_bvp_time_diff = np.diff(frame_timestamps)
        # resampled_spo2_time_diff = np.diff(frame_timestamps)
        # resampled_rr_time_diff = np.diff(frame_timestamps)

        # # Calculate the sampling frequency for each resampled signal (assuming uniform resampling)
        # resampled_bvp_sampling_rate = 1 / np.mean(resampled_bvp_time_diff)
        # resampled_spo2_sampling_rate = 1 / np.mean(resampled_spo2_time_diff)
        # resampled_rr_sampling_rate = 1 / np.mean(resampled_rr_time_diff)

        # print(f"Resampled BVP sampling frequency: {resampled_bvp_sampling_rate} Hz")
        # print(f"Resampled SpO2 sampling frequency: {resampled_spo2_sampling_rate} Hz")
        # print(f"Resampled RR sampling frequency: {resampled_rr_sampling_rate} Hz")
        # Process frames, BVP signals, and SpO2 signals according to the configuration
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            rr = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS, min_freq=0.1, max_freq = 0.5)
        else:
            bvps = resampled_bvp
            spo2 = resampled_spo2
            rr = resampled_rr
        # plot_rr_wave_(rr)
        # Label once here
        # print("video_file:")
        # print(video_file)

        # if "RGB_H264" in video_file:
        if experiment_id=='Camera2':
            print("experiment_id=='Camera2'")
            print(f"video_file: {video_file}")
            frames_clips, bvps_clips, spo2_clips, rr_clips, confidences_clips = self.preprocess(frames, bvps, spo2, rr, config_preprocess, "face", name_id, subject_id, experiment_id)
            print(f"face Frames clips shape: {frames_clips.shape}")
            print(f"BVP clips shape: {bvps_clips.shape}")
            print(f"SpO2 clips shape: {spo2_clips.shape}")
            print(f"RR clips shape: {rr_clips.shape}")
            print(f"confidences_clips shape: {confidences_clips.shape}")
            
            filename = f"{name_id}_{subject_id}_{experiment_id}"
            input_name_list, label_name_list, spo2_name_list, rr_name_list, confidence_path_name_list = self.save_multi_process(frames_clips, bvps_clips, spo2_clips, rr_clips, confidences_clips, filename)
            file_list_dict[i] = input_name_list
        else:
            print("experiment_id=='Camera1'")
            frames_clips, _, _, _, confidences_clips = self.preprocess(frames, None, None, None, config_preprocess, "face_IR", name_id, subject_id, experiment_id) 
            # print(f"finger Frames clips shape: {frames_clips.shape}"
            # print(f"face Frames clips shape: {frames_clips.shape}")
            # print(f"BVP clips shape: {bvps_clips.shape}")
            # print(f"SpO2 clips shape: {spo2_clips.shape}")
            # print(f"RR clips shape: {rr_clips.shape}")            
            filename = f"{name_id}_{subject_id}_{experiment_id}_IR"
            input_name_list, confidence_path_name_list = self.save_multi_process_no_labels(frames_clips, confidences_clips, filename)
            file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """Load preprocessed data listed in the file list."""
        type_info = self.info.TYPE
        state = self.info.STATE
        # print(f"self.info: {self.info}")
        # print(f"type_info: {type_info}, state: {state}")
        
        file_list_path = self.file_list_path   # Get file list path
        print("11111file_list_path:",file_list_path)
        file_list_df = pd.read_csv(file_list_path)  # Read file list
        inputs_temp = file_list_df['input_files'].tolist()  # Get input file list
        inputs_face = [] 
        inputs_face_IR = [] 
        # v01 v02 v03 v04 face finger configuration information
        for each_input in inputs_temp:
            #print(f"each_input: {each_input}")
            info = each_input.split(os.sep)[-1].split('_')
            # print("info=",info)
            state = int(info[2][-1])
            # print("state=",state)
            #print(f"state: {state}")
            if info[3] == "Camera1":   # face IR
                type = 1
            else:
                type = 2
            #print(f"info:{info}, state: {state}, type: {type}")
            # Filter data according to configuration information

            if (state in self.info.STATE) and (type in self.info.TYPE) and type == 2:
                inputs_face.append(each_input)
                # print("state=",state)
                # print(f"each_input: {each_input}")
            # finger 2
            if (state in self.info.STATE) and (type in self.info.TYPE) and type == 1:
                inputs_face_IR.append(each_input)
        print(f"inputs_face_len: {len(inputs_face)}, inputs_face_IR: {len(inputs_face_IR)}")
        if not inputs_face and not inputs_face_IR:
            raise ValueError(self.dataset_name + ' Dataset loading error!')
        # print(inputs_face)
        if not inputs_face:
            raise ValueError(self.dataset_name + ' Dataset loading error!')
        # single face finger both
        # print(f"len(type_info): {len(type_info)}") 
        if len(type_info) == 1:
            if type_info[0] == 1: # face
                inputs_face = sorted(inputs_face)
                labels_bvp = [input_file.replace("face_input", "hr") for input_file in inputs_face]  
                labels_spo2 = [input_file.replace("face_input", "spo2") for input_file in inputs_face]
                labels_rr = [input_file.replace("face_input", "rr") for input_file in inputs_face]  
                self.inputs = inputs_face
                self.labels_bvp = labels_bvp
                self.labels_spo2 = labels_spo2
                self.labels_rr = labels_rr
                self.preprocessed_data_len = len(inputs_face)   
            else: # finger    
                inputs_face_IR = sorted(inputs_face_IR)
                labels_bvp = [input_file.replace("finger_input", "hr") for input_file in inputs_face_IR]  
                labels_spo2 = [input_file.replace("finger_input", "spo2") for input_file in inputs_face_IR]  
                labels_rr = [input_file.replace("finger_input", "rr") for input_file in inputs_face_IR]
                self.inputs_face_IR = inputs_face_IR
                self.labels_bvp = labels_bvp
                self.labels_spo2 = labels_spo2
                self.labels_rr = labels_rr
                self.preprocessed_data_len = len(inputs_face_IR)   
        else:
            inputs_face = sorted(inputs_face)
            inputs_face_IR = sorted(inputs_face_IR)
            labels_bvp = [input_file.replace("face_input", "hr") for input_file in inputs_face]  
            labels_spo2 = [input_file.replace("face_input", "spo2") for input_file in inputs_face]
            labels_rr = [input_file.replace("face_input", "rr") for input_file in inputs_face]  
            self.inputs = inputs_face
            self.inputs_face_IR = inputs_face_IR
            self.labels_bvp = labels_bvp
            self.labels_spo2 = labels_spo2
            self.labels_rr = labels_rr
            # Mixed training also only requires one of the lengths
            self.preprocessed_data_len = len(inputs_face)   
            print(f"inputs_face: {inputs_face[20]}")
            print(f"inputs_face_IR: {inputs_face_IR[20]}")
            print(f"labels_bvp: {labels_bvp[20]}")
            print(f"labels_spo2: {labels_spo2[20]}")
            print(f"labels_rr: {labels_rr[20]}")
            

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
    def read_rr(rr_file):
        """Reads a respiratory rate (RR) signal file with timestamps."""
        data = pd.read_csv(rr_file)
        timestamps = data['timestamp'].values
        rr_values = data['resp'].values
        return timestamps, rr_values

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

    def save_comparison_images(self, frames, confidences, name_id, subject_id, experiment_id):
        # 采样28张图片
        num_frames = len(frames)
        sample_indices = []
        
        # 每隔128帧采样，最多采28个样本
        for i in range(28):
            index = i * 128
            if index < num_frames:
                sample_indices.append(index)
            else:
                sample_indices.append(num_frames - 1)  # 如果没有足够的帧，则用最后一帧
        
        # 设置4行7列的子图布局
        fig, axes = plt.subplots(4, 7, figsize=(14, 8))
        
        # 遍历并绘制每一张图片
        for i, ax in enumerate(axes.flat):
            index = sample_indices[i]
            
            # 获取对应帧和置信度
            frame = frames[index]
            confidence = confidences[index]
            
            # 确保图像是8位无符号整数类型
            if frame.dtype == 'float64':
                frame = cv2.convertScaleAbs(frame)

            # 将图像转为RGB格式，matplotlib需要RGB而非BGR
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Confidence: {confidence:.2f}')
            ax.axis('off')

        # 调整子图之间的间距
        plt.tight_layout()

        # 存储图像
        comparison_image_path = f"/data01/jz/rppg_tool_HMS/result/frame_crop_RF/{name_id}_{subject_id}_{experiment_id}_frame_comparison.png"
        plt.savefig(comparison_image_path)
        plt.close()


    def preprocess(self, frames, bvps, spo2, rr, config_preprocess, video_type, name_id, subject_id, experiment_id):
        """Preprocesses a pair of data."""
        DO_CROP_FACE = True
        # 获取第一帧用于保存裁剪前后的图像
        # first_frame = frames[0]
        # f150_frame = frames[150]

        frames, confidences = self.crop_face_resize(
            frames,
            DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        
        # 调用 save_comparison_images 函数，保存裁剪前后的对比图
        self.save_comparison_images(frames, confidences, name_id, subject_id, experiment_id)


        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)

        if bvps is not None and spo2 is not None and rr is not None:
            if config_preprocess.LABEL_TYPE == "Raw":
                pass
            elif config_preprocess.LABEL_TYPE == "DiffNormalized":
                bvps = BaseLoader.diff_normalize_label(bvps)
                rr = BaseLoader.diff_normalize_label(rr)
            elif config_preprocess.LABEL_TYPE == "Standardized":
                bvps = BaseLoader.standardized_label(bvps)
                rr = BaseLoader.standardized_label(rr)
            else:
                raise ValueError("Unsupported label type!")
 
            if config_preprocess.DO_CHUNK:
                frames_clips, bvps_clips, spo2_clips, rr_clips, confidences_clips = self.chunk(data, bvps, spo2, rr, confidences, config_preprocess.CHUNK_LENGTH, video_type)
            else:
                frames_clips = np.array([data])
                bvps_clips = np.array([bvps])
                spo2_clips = np.array([spo2])
                rr_clips = np.array([rr])

            return frames_clips, bvps_clips, spo2_clips, rr_clips, confidences_clips
        else:
            if config_preprocess.DO_CHUNK:
                frames_clips, _, _, _, confidences_clips = self.chunk(data, None, None, None, confidences, config_preprocess.CHUNK_LENGTH, video_type)
            else:
                frames_clips = np.array([data])

            return frames_clips, None, None, None, confidences_clips

    def save_multi_process_no_labels(self, frames_clips, confidence_clips, filename):
        """Save all the chunked data with multi-thread processing (no labels for finger)."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        count = 0
        input_path_name_list = []
        confidence_path_name_list = []  # 新增置信度路径列表

        for i in range(len(frames_clips)):
            input_path_name = os.path.join(self.cached_path, f"{filename}_input_{count}.npy")
            confidence_path_name = os.path.join(self.cached_path, f"{filename}_confidence_{count}.npy")  # 置信度文件路径
            
            input_path_name_list.append(input_path_name)
            confidence_path_name_list.append(confidence_path_name)  # 添加到列表
            
            np.save(input_path_name, frames_clips[i])
            np.save(confidence_path_name, confidence_clips[i])  # 保存置信度数据
            
            count += 1

        return input_path_name_list, confidence_path_name_list  # 返回两个列表


def plot_all(rr, rr_re):
 
# # plot the filtered RR data and the power spectral density
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    axes[0].plot(rr, label='RR Data')
    axes[0].set_title('RR Data Plot')
    axes[0].set_xlabel('data Number')
    axes[0].set_ylabel('Data Value')
    axes[0].legend()

    # # plot the filtered RR data and the power spectral density
    axes[1].plot(rr_re, label='RR sample Data')
    axes[1].set_title('RR Data Plot')
    axes[1].set_xlabel('data Number')
    axes[1].set_ylabel('Data Value')
    axes[1].legend()

    plt.savefig('./rr_re.png')
    plt.show()

def plot_rr_std(rr_label):

    # rr_pred = np.array(rr_pred)
    # rr_label = rr_label.flatten()
    # rr_label = np.array(rr_label)
    # Plotting
    plt.figure(figsize=(20, 6))
    #plt.plot(rr_pred, label="Predicted RR", color='blue', linewidth=1.5)
    plt.plot(rr_label, label="True RR", color='red', linewidth=1.5)
    
    plt.title("RR ")
    plt.xlabel("number")
    plt.ylabel("RR Rate")
    plt.legend()
    plt.savefig('./std.png')
    plt.show()

def plot_rr_wave_(rr_label):

    # rr_pred = np.array(rr_pred)
    rr_label = rr_label.flatten()
    rr_label = np.array(rr_label)
    # Plotting
    plt.figure(figsize=(20, 6))
    #plt.plot(rr_pred, label="Predicted RR", color='blue', linewidth=1.5)
    plt.plot(rr_label, label="True RR", color='red', linewidth=1.5)
    
    plt.title("RR ")
    plt.xlabel("number")
    plt.ylabel("RR Rate")
    plt.legend()
    plt.savefig('./3333.png')
    plt.show()
