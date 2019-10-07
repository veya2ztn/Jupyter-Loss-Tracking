import torch
import torch.nn as nn
import numpy as np

import copy
import torch.nn.functional as F


#######################################
#######  LOG_RECORDER   ###############
#######################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.win100=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #self.avg = self.sum / self.count
        self.win100.append(val)
        if len(self.win100) > 100:_=self.win100.pop(0)
        sum100=sum(self.win100)
        self.avg=1.0*sum100/len(self.win100)
        self.output = self.avg

class IdentyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val, n=1):
        self.val    = val
        self.output = self.val

class Curve_data(object):
    def __init__(self,max_leng = np.inf,Reduce=False):
        self.x = []
        self.y = []
        self.ml= max_leng
        if max_leng is np.inf:
            self.bound_x = None
            self.bound_y = None
            Reduce=False
        else:
            self.bound_x = [0,max_leng]
            self.bound_y = None
        self.reduce = Reduce
        self.reducer= 100
        self.reduce_limit = 10000

    @property
    def data(self):
        return [self.x,self.y]

    def reset(self):
        '''
        determinate the range of y via previous data
        '''
        y_win    = max(self.y)-min(self.y)
        now_at   = self.y[-1]
        self.y_bounds = [now_at-y_win, now_at+0.2*y_win]
        self.x = []
        self.y = []

    def reduce(self):
        if not self.reduce:return
        selector = np.linspace(0,self.reduce_limit,num=self.reducer).astype('int')
        self.x=self.x[selector]
        self.y=self.y[selector]

    def add_xy(self,xy):
        if len(self.x) > self.ml:self.reset()
        if len(self.x) > self.reduce_limit:self.reduce()
        self.x.append(xy[0])
        self.y.append(xy[1])

    def add_y(self,y):
        x = 0 if len(self.x)==0 else (self.x[-1]+1)%(self.ml+1)
        self.add_xy([x,y])

class RecordLoss:

    def __init__(self,loss_records=None,graph_set=None):
        self.loss_records=loss_records
        self.graph_set = graph_set
        self.initialQ  = False
        self.global_mode = False

    def initial(self,num):
        if self.loss_records is None:self.loss_records = [AverageMeter() for i in range(num)]
        assert len(self.loss_records) == num
        if self.graph_set is None:self.graph_set = [Curve_data() for i in range(num)]
        assert len(self.graph_set) == num
        self.initialQ  = True

    def record_loss(self,loss_recorder):
        return loss_recorder.output

    def update_record(self,recorder,loss):
        if isinstance(loss,torch.Tensor):loss = loss.item()
        recorder.update(loss)

    def update(self,loss_list):
        if not self.initialQ: self.initial(len(loss_list))
        for loss_recorder,g,loss in zip(self.loss_records,self.graph_set,loss_list):
            self.update_record(loss_recorder,loss)# record data in recorder
            loss = self.record_loss(loss_recorder)# record data until get enough for next statistic result
            if loss:g.add_y(loss)

    def step(self,step,loss_list):
        self.update(loss_list)

    def update_graph(self,mb,step):
        graphs  =[[g.data]  for g in self.graph_set]
        x_bounds=[g.bound_x for g in self.graph_set]
        y_bounds=[g.bound_y for g in self.graph_set]
        mb.update_graph_multiply(graphs,x_bounds,y_bounds)
        #mb.update_graph(graphs, x_bounds, y_bounds)
            #return ll

    def print2file(self,step,file_name):
        with open(file_name,'a') as log_file:
            ll=["{:.4f}".format(self.record_loss(recorder)) for recorder in self.loss_records]
            printedstring=str(step)+' '+' '.join(ll)+'\n'
            #print(printedstring)
            _=log_file.write(printedstring)
