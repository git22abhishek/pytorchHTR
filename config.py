class Data:
    DATASET_ROOT_DIR = '../datasets/IAM'
    CSV_FILE_PATH = '../datasets/IAM/IAM_df.csv'
    SHUFFLE = True
    SEED = 42
    NUM_WORKERS = 0
    DEFAULT_PARTITION = False
    PARTITION = (0.6, 0.2, 0.2)
    BATCH_SIZE = (32, 16, 16)

    def _asdict():
        return {
            'DATASET_ROOT_DIR': Data.DATASET_ROOT_DIR,
            'CSV_FILE_PATH': Data.CSV_FILE_PATH,
            'SHUFFLE': Data.SHUFFLE,
            'SEED': Data.SEED,
            'NUM_WORKERS': Data.NUM_WORKERS,
            'DEFAULT_PARTITION': Data.DEFAULT_PARTITION,
            'PARTITION': Data.PARTITION,
            'BATCH_SIZE': Data.BATCH_SIZE
        }


class Optimizer:
    LR = 0.003

    def _asdict():
        return {'LR': Optimizer.LR}


class Scheduler:
    FACTOR = 0.1
    PATIENCE = 5
    VERBOSE = True

    def _asdict():
        return {
            'FACTOR': Scheduler.FACTOR,
            'PATIENCE': Scheduler.PATIENCE,
            'VERBOSE': Scheduler.VERBOSE,

        }


class CTCLoss:
    BLANK = 0
    REDUCTION = 'sum'
    ZERO_INFINITY = True

    def _asdict():
        return {
            'BLANK': CTCLoss.BLANK,
            'REDUCTION': CTCLoss.REDUCTION,
            'ZERO_INFINITY': CTCLoss.ZERO_INFINITY,
        }


class config:
    data = Data()
    opt = Optimizer()
    sch = Scheduler()
    ctc_loss = CTCLoss()
    TIME_STEPS = 100
    NUM_EPOCHS = 20
    RESUME_TRAINING = False
    TRAIN_CHECKPOINT_PATH = '../checkpoints/training_state.pth'

    def _asdict():
        return {
            'data': Data._asdict(),
            'opt': Optimizer._asdict(),
            'sch': Scheduler._asdict(),
            'ctc_loss': CTCLoss._asdict(),
            'TIME_STEPS': config.TIME_STEPS,
            'NUM_EPOCHS': config.NUM_EPOCHS,
            'RESUME_TRAINING': config.RESUME_TRAINING,
            'TRAIN_CHECKPOINT_PATH': config.TRAIN_CHECKPOINT_PATH,
        }
