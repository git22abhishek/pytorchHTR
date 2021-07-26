class Data:

    def __init__(self) -> None:

        self.DATASET_ROOT_DIR = '../datasets/IAM'
        self.CSV_FILE_PATH = '../datasets/IAM/IAM_df.csv'
        self.SHUFFLE = True
        self.SEED = 42
        self.NUM_WORKERS = 0
        self.DEFAULT_PARTITION = False
        self.PARTITION = (0.6, 0.2, 0.2)
        self.BATCH_SIZE = (32, 16, 16)

    def _asdict(self):

        return {
            'DATASET_ROOT_DIR': self.DATASET_ROOT_DIR,
            'CSV_FILE_PATH': self.CSV_FILE_PATH,
            'SHUFFLE': self.SHUFFLE,
            'SEED': self.SEED,
            'NUM_WORKERS': self.NUM_WORKERS,
            'DEFAULT_PARTITION': self.DEFAULT_PARTITION,
            'PARTITION': self.PARTITION,
            'BATCH_SIZE': self.BATCH_SIZE
        }


class Optimizer:

    def __init__(self) -> None:
        self.LR = 0.003

    def _asdict(self):
        return {'LR': self.LR}


class Scheduler:

    def __init__(self) -> None:

        self.FACTOR = 0.1
        self.PATIENCE = 5
        self.VERBOSE = True

    def _asdict(self):

        return {
            'FACTOR': self.FACTOR,
            'PATIENCE': self.PATIENCE,
            'VERBOSE': self.VERBOSE,
        }


class CTCLoss:

    def __init__(self) -> None:

        self.BLANK = 0
        self.REDUCTION = 'sum'
        self.ZERO_INFINITY = True

    def _asdict(self):

        return {
            'BLANK': self.BLANK,
            'REDUCTION': self.REDUCTION,
            'ZERO_INFINITY': self.ZERO_INFINITY,
        }


class Config:

    def __init__(self) -> None:

        self.data = Data()
        self.opt = Optimizer()
        self.sch = Scheduler()
        self.ctc_loss = CTCLoss()
        self.TIME_STEPS = 100
        self.NUM_EPOCHS = 20
        self.RESUME_TRAINING = False
        self.TRAIN_CHECKPOINT_PATH = '../checkpoints/training_state.pth'

    def _asdict(self):

        return {
            'data': self.data._asdict(),
            'opt': self.opt._asdict(),
            'sch': self.sch._asdict(),
            'ctc_loss': self.ctc_loss._asdict(),
            'TIME_STEPS': self.TIME_STEPS,
            'NUM_EPOCHS': self.NUM_EPOCHS,
            'RESUME_TRAINING': self.RESUME_TRAINING,
            'TRAIN_CHECKPOINT_PATH': self.TRAIN_CHECKPOINT_PATH,
        }
