import datetime
import os
import sys
import shutil
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GLOG_minloglevel"] ="3"

def git_root(path):
    current_path = os.path.abspath(path)
    while current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    return None

def set_root():
    current = os.getcwd()
    sys.path.append(git_root(current))

set_root()
import QoL as qol
config = qol.device_configuration()

def proc(*args):
    path, lock = args
    ins = qol.mediapipe_landmarks(path)
    with lock:
        ins.to_csv(normalize=True)
        qol.dump_object(ins, 'dump.pkl')

def main(paths):
    with mp.Manager() as manager:
        lock = manager.Lock()
        with mp.Pool(processes=config.max_cores//2) as pool:
            pool.starmap(proc, [(path, lock) for path in paths])
            pool.close()
            pool.join()
    shutil.move('dump.pkl', r'graph-processing/')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    _, _, paths = qol.retrieve_raw_paths()
    tiempo_0 = datetime.datetime.today()
    
    print('Procesamiento iniciado -',datetime.datetime.today())
    main(paths)
    print('Procesamiento terminado -', datetime.datetime.today())
    print('Tiempo invertido: ',datetime.datetime.today()-tiempo_0)