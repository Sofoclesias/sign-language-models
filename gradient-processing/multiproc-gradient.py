import datetime
import warnings
import QoL as qol
warnings.filterwarnings('ignore')

def proc(path):
    ins = qol.hog_transform(path)
    ins.to_csv(normalize=True)
    qol.dump_object(ins,'dump.pkl')

if __name__ == '__main__':
    # Iniciar PyTorch con configuraciones
    config = qol.device_configuration()

    if config.processing_unit == 'cuda':
        import torch.multiprocessing as multiproc
        try:
            multiproc.set_start_method('spawn')
        except RuntimeError:
            pass
    else:
        import multiprocessing as multiproc

    train, test, _all = qol.retrieve_raw_paths()

    tiempo_0 = datetime.datetime.today()
    print('Procesamiento iniciado -',datetime.datetime.today())

    with multiproc.Pool(processes=config.max_cores//2) as pool:
        pool.map(proc, _all)
        pool.close()
        pool.join()

    print('Procesamiento termina -', datetime.datetime.today())
    print('Tiempo invertido: ',datetime.datetime.today()-tiempo_0)
    