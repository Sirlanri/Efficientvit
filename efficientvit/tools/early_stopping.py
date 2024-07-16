import numpy as np
from configs.seg.train_seg_configs import *

#早停机制
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.lossArray:list[float]=[]

    #检验loss数组的数据是否平滑
    def isSmooth(self) -> bool:
        stdnum=np.std(self.lossArray)
        if stdnum<Std_Smooth:
            return True
        else:
            return False

    def __call__(self, val_loss):

        score = -val_loss

        if len(self.lossArray)>=self.patience:
            self.lossArray.pop(0)
        self.lossArray.append(val_loss)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Smooth= {np.std(self.lossArray)}')
            if self.counter >= self.patience and self.isSmooth():
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0