import datetime
import warnings
import QOL_funcs as qol
warnings.filterwarnings('ignore')

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

    _all = [(image_path, False, True) for image_path in qol.all]
    print('Procesamiento iniciado - ', datetime.datetime.today())

    with multiproc.Pool(processes=config.max_cores//2) as pool:
        pool.starmap(qol.mediapipe_landmarks, _all)
        pool.close()
        pool.join()

    print('Procesamiento terminado - ', datetime.datetime.today())