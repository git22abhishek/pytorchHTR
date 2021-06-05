class Data:
    DATASET_ROOT_DIR = '/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed'
    CSV_FILE_PATH = 'IAM_df.csv'
    SHUFFLE = True
    SEED = 42
    NUM_WORKERS = 2
    DEFAULT_PARTITION = False
    PARTITION = (0.006, 0.002, 0.002)
    BATCH_SIZE = (2, 1, 1)


class Optimizer:
    LR = 0.001


class Scheduler:
    FACTOR = 0.1
    PATIENCE = 5
    VERBOSE = True


class CTCLoss:
    BLANK = 0
    REDUCTION = 'sum'
    ZERO_INFINITY = True


class config:
    data = Data()
    opt = Optimizer()
    sch = Scheduler()
    ctc_loss = CTCLoss()
    TIME_STEPS = 100
    NUM_EPOCHS = 1
    RESUME_TRAINING = False
    TRAIN_CHECKPOINT_PATH = 'checkpoints/training_state.pth'
