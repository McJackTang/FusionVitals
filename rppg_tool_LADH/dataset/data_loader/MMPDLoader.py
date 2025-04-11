""" The dataloader for MMPD datasets.

"""

import os
import cv2
import glob
import numpy as np
import re

# 从基类加载器导入
from .BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import scipy.io as sio
import sys
import itertools
from warnings import simplefilter
from scipy.signal import butter, filtfilt

# 忽略未来警告
simplefilter(action='ignore', category=FutureWarning)

class MMPDLoader(BaseLoader):
    """MMPD数据集的数据加载器。"""

    def __init__(self, name, data_path, config_data):
        """
        初始化MMPD数据加载器。
        参数:
            name (str): 数据加载器的名称。
            data_path (str): 存储原始视频和bvp数据的文件夹路径。
                             例如，data_path应该是"mat_dataset"，其目录结构如下：
                             -----------------
                             mat_dataset/
                             |   |-- subject1/
                             |       |-- p1_0.mat
                             |       |-- p1_1.mat
                             |       |...
                             |   |-- subject2/
                             |       |-- p2_0.mat
                             |       |-- p2_1.mat
                             |       |...
                             |...
                             |   |-- subjectn/
                             |       |-- pn_0.mat
                             |       |...
                             -----------------
            config_data (CfgNode): 数据配置（参考config.py）。
        """
        self.info = config_data.INFO  # 存储配置信息
        super().__init__(name, data_path, config_data)  # 调用父类构造函数

    def get_raw_data(self, raw_data_path):
        """
        返回路径下的数据目录（适用于MMPD数据集）。
        参数:
            raw_data_path (str): 原始数据的路径。
        返回:
            dirs (list): 包含数据路径和其他信息的字典列表。
        """
        
        # 获取所有符合“subject*”模式的文件夹路径
        data_dirs = glob.glob(raw_data_path + os.sep + 'subject*')
        if not data_dirs:
            raise ValueError(self.dataset_name + ' 数据路径为空！')
        dirs = list()
        for data_dir in data_dirs:
            subject = int(os.path.split(data_dir)[-1][7:])  # 获号取受试者编
            mat_dirs = os.listdir(data_dir)  # 获取受试者文件夹下的所有文件
            for mat_dir in mat_dirs:
                index = mat_dir.split('_')[-1].split('.')[0]  # 获取文件的索引
                dirs.append({'index': index, 
                             'path': data_dir + os.sep + mat_dir,  # 完整的文件路径
                             'subject': subject})  # 受试者编号
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """
        返回按begin和end值划分的数据目录子集，确保分割之间没有重叠的受试者。
        参数:
            data_dirs (list): 包含数据路径和其他信息的字典列表。
            begin (float): 数据集的开始比例。
            end (float): 数据集的结束比例。
        返回:
            data_dirs_new (list): 分割后的数据目录子集。
        """

        # 如果begin和end分别为0和1，则返回完整目录
        if begin == 0 and end == 1:
            return data_dirs
        
        data_info = dict()
        for data in data_dirs:
            index = data['index']
            data_dir = data['path']
            subject = data['subject']
            # 创建一个以受试者编号为索引的数据目录字典
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = list(data_info.keys())  # 获取所有受试者编号
        subj_list = sorted(subj_list)  # 对受试者编号进行排序
        num_subjs = len(subj_list)  # 受试者总数

        # 获取数据集的分割（根据开始/结束比例）
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('用于分割的受试者ID：', [subj_list[i] for i in subj_range])

        # 将符合分割范围的文件路径添加到新列表中
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        由preprocess_dataset调用的多进程子程序。
        参数:
            data_dirs (list): 包含数据路径和其他信息的字典列表。
            config_preprocess (CfgNode): 预处理配置。
            i (int): 当前进程索引。
            file_list_dict (dict): 存储文件列表的字典。
        """
        frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = self.read_mat(data_dirs[i]['path'])

        # 构建保存文件的名称
        saved_filename = 'subject' + str(data_dirs[i]['subject'])
        saved_filename += f'_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}'

        frames = (np.round(frames * 255)).astype(np.uint8)  # 将帧数据缩放到0-255
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)  # 重采样PPG信号
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)  # 预处理数据

        # 保存预处理后的数据并更新文件列表
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def read_mat(self, mat_file):
        """
        读取.mat文件并返回其中的数据。
        参数:
            mat_file (str): .mat文件的路径。
        返回:
            frames (np.ndarray): 视频帧数据。
            bvps (np.ndarray): BVP信号。
            light (int): 光照条件。
            motion (int): 运动状态。
            exercise (int): 运动情况。
            skin_color (int): 肤色。
            gender (int): 性别。
            glasser (int): 是否戴眼镜。
            hair_cover (int): 是否遮住头发。
            makeup (int): 是否化妆。
        """
        try:
            mat = sio.loadmat(mat_file)  # 读取.mat文件
        except:
            for _ in range(20):
                print(mat_file)
        frames = np.array(mat['video'])  # 获取视频帧数据

        # 根据配置决定是否使用伪PPG标签
        if self.config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else: 
            bvps = np.array(mat['GT_ppg']).T.reshape(-1)  # 获取真实的PPG信号

        # 获取其他信息
        light = mat['light']
        motion = mat['motion']
        exercise = mat['exercise']
        skin_color = mat['skin_color']
        gender = mat['gender']
        glasser = mat['glasser']
        hair_cover = mat['hair_cover']
        makeup = mat['makeup']
        information = [light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup]
        
        # 转换信息为数值
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = self.get_information(information)

        return frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
    
    def load_preprocessed_data(self):
        """
        加载文件列表中列出的预处理数据。
        参数:
            无
        返回:
            无
        """
        file_list_path = self.file_list_path  # 获取文件列表路径
        file_list_df = pd.read_csv(file_list_path)  # 读取文件列表
        inputs_temp = file_list_df['input_files'].tolist()  # 获取输入文件列表
        inputs = []
        for each_input in inputs_temp:
            info = each_input.split(os.sep)[-1].split('_')
            light = int(info[1][-1])
            motion = int(info[2][-1])
            exercise = int(info[3][-1])
            skin_color = int(info[4][-1])
            gender = int(info[5][-1])
            glasser = int(info[6][-1])
            hair_cover = int(info[7][-1])
            makeup = int(info[8][-1])
            # 根据配置信息过滤数据
            if (light in self.info.LIGHT) and (motion in self.info.MOTION) and \
                (exercise in self.info.EXERCISE) and (skin_color in self.info.SKIN_COLOR) and \
                (gender in self.info.GENDER) and (glasser in self.info.GLASSER) and \
                (hair_cover in self.info.HAIR_COVER) and (makeup in self.info.MAKEUP):
                inputs.append(each_input)
        if not inputs:
            raise ValueError(self.dataset_name + ' 数据集加载错误！')
        inputs = sorted(inputs)  # 对输入文件名列表进行排序
        labels = [input_file.replace("input", "label") for input_file in inputs]  # 获取对应的标签文件
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def get_information(information):
        """
        根据提供的信息返回数值标签。
        参数:
            information (list): 包含各类信息的列表。
        返回:
            light (int): 光照条件。
            motion (int): 运动状态。
            exercise (int): 运动情况。
            skin_color (int): 肤色。
            gender (int): 性别。
            glasser (int): 是否戴眼镜。
            hair_cover (int): 是否遮住头发。
            makeup (int): 是否化妆。
        """
        # 将光照条件转换为数值标签
        light = ''
        if information[0] == 'LED-low':
            light = 1
        elif information[0] == 'LED-high':
            light = 2
        elif information[0] == 'Incandescent':
            light = 3
        elif information[0] == 'Nature':
            light = 4
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下光照标签: {0}".format(information[0]))

        # 将运动状态转换为数值标签
        motion = ''
        if information[1] == 'Stationary' or information[1] == 'Stationary (after exercise)':
            motion = 1
        elif information[1] == 'Rotation':
            motion = 2
        elif information[1] == 'Talking':
            motion = 3
        # 'Watching Videos'是MMPD数据集的错误标签，应处理为'Walking'
        elif information[1] == 'Walking' or information[1] == 'Watching Videos':
            motion = 4
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下运动标签: {0}".format(information[1]))
        
        # 将运动情况转换为数值标签
        exercise = ''
        if information[2] == 'True':
            exercise = 1
        elif information[2] == 'False':
            exercise = 2
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下运动情况标签: {0}".format(information[2]))

        # 获取肤色数值
        skin_color = information[3][0][0]
        if skin_color != 3 and skin_color != 4 and skin_color != 5 and skin_color != 6:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下肤色标签: {0}".format(information[3][0][0]))

        # 将性别转换为数值标签
        gender = ''
        if information[4] == 'male':
            gender = 1
        elif information[4] == 'female':
            gender = 2
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下性别标签: {0}".format(information[4]))

        # 将是否戴眼镜转换为数值标签
        glasser = ''
        if information[5] == 'True':
            glasser = 1
        elif information[5] == 'False':
            glasser = 2
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下眼镜标签: {0}".format(information[5]))

        # 将是否遮住头发转换为数值标签
        hair_cover = ''
        if information[6] == 'True':
            hair_cover = 1
        elif information[6] == 'False':
            hair_cover = 2
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下头发覆盖标签: {0}".format(information[6]))
        
        # 将是否化妆转换为数值标签
        makeup = ''
        if information[7] == 'True':
            makeup = 1
        elif information[7] == 'False':
            makeup = 2
        else:
            raise ValueError("MMPD或Mini-MMPD数据集标签错误！不支持以下化妆标签: {0}".format(information[7]))

        return light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
