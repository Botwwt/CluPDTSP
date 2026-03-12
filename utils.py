"""
The MIT License

"""

import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import pytz
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # ================================================================= #
    # BUG FIX: Add encoding='utf-8' to handle non-ASCII characters in paths.
    # ================================================================= #
    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode, encoding='utf-8')
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

# The remaining utility classes below are unchanged.
# This file is kept complete; the logger update is limited to create_logger.
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count else 0
class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}
    def get_raw_data(self):
        return self.keys, self.data
    def set_raw_data(self, r_data):
        self.keys, self.data = r_data
    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key): value = [len(self.data[key]), args]
                else: value = [0, args]
            elif type(args) == tuple: value = list(args)
            elif type(args) == list: value = args
            else: raise ValueError('Unsupported value type')
        elif len(args) == 2: value = [args[0], args[1]]
        else: raise ValueError('Unsupported value type')
        if key in self.keys: self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)
    def has_key(self, key):
        return key in self.keys
    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)
        return split[1].squeeze().tolist()
    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)
        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()
        if type(xs) is not list: return xs, ys
        if start_idx == 0: return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else: raise KeyError('no start_idx value in X axis data.')
    def get_keys(self):
        return self.keys
class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0
    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1
    def get_est_string(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)
        elapsed_time_str = "{:.2f}h".format(elapsed_time/3600) if elapsed_time > 3600 else "{:.2f}m".format(elapsed_time/60)
        remain_time_str = "{:.2f}h".format(remain_time/3600) if remain_time > 3600 else "{:.2f}m".format(remain_time/60)
        return elapsed_time_str, remain_time_str
def util_print_log_array(logger, result_log: LogData):
    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))
def util_save_log_image_with_label(result_file_prefix, img_params, result_log: LogData, labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    _build_log_image_plt(img_params, result_log, labels)
    if labels is None: labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)
def _build_log_image_plt(img_params, result_log: LogData, labels=None):
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)
    with open(log_image_config_file, 'r') as f:
        config = json.load(f)
    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)
    if labels is None: labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)
    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None: ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None: ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)
    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None: xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None: xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)
    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])
def copy_all_src(dst_root):
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))
    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else: home_dir = tmp_dir1
    dst_path = os.path.join(dst_root, 'src')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for key, value in list(sys.modules.items()):
        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)
            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))
                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0
                    while os.path.exists(filepath.format(post_index)): post_index += 1
                    dst_filepath = filepath.format(post_index)
                import re
                if re.search(r'_.*\.py$', src_abspath): continue
                shutil.copy(src_abspath, dst_filepath)
