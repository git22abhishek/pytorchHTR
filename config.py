class Data:
    def __init__(self):
        self.DATASET_ROOT_DIR = '/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed'
        self.CSV_FILE_PATH = 'IAM_df.csv'
        self.SHUFFLE = True
        self.SEED = 42
        self.NUM_WORKERS = 0
        self.DEFAULT_SPLIT = False
        self.SPLIT = (0.6, 0.2, 0.2)
        self.BATCH_SIZE = (32, 16, 16)


class Optimizer:
    def __init__(self):
        self.LR = 0.003


class Scheduler:
    def __init__(self):
        self.FACTOR = 0.1
        self.PATIENCE = 5
        self.VERBOSE = True


class CTCLoss:
    def __init__(self):
        self.BLANK = 0
        self.REDUCTION = 'sum'
        self.ZERO_INFINITY = True


class config:
    def __init__(self):
        self.data = Data()
        self.opt = Optimizer()
        self.sch = Scheduler()
        self.ctc_loss = CTCLoss()
        self.TIME_STEPS = 100
        self.NUM_EPOCHS = 20
        self.RESUME_TRAINING = False
        self.TRAIN_CHECKPOINT_PATH = 'checkpoints/training_state.pth'
