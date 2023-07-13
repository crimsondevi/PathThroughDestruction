import torch.nn as nn
import torch.optim as optim
from operator import attrgetter

class DataLoaders:
    def __init__(self, *dls): self.train,self.valid = dls[:2]

class Callback(): order = 0

def run_cbs(cbs, method_nm, learner):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learner)

class Learner():
    def __init__(self, model, dls=(0,), loss_func=nn.MSELoss(), lr=0.003, cbs=[]):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.lr = lr
        self.cbs = cbs

        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.sched = None
            
    def predict(self):
        inputs, _ = self.batch
        self.preds = self.model(inputs)
    
    def get_loss(self):
        _, labels = self.batch
        self.loss = self.loss_func(self.preds, labels)

    def step(self):
        self.optimizer.step()
        if self.sched:
            self.sched.step()
        
    def _one_batch(self, train):
        self.predict()
        self.get_loss()
        if train:
            self.loss.backward()
            self.step()        
            self.optimizer.zero_grad()
    
    def _one_epoch(self, train):
        self.train = train
        self.model.train()
        self.dl = self.dls.train if train else self.dls.train
        for self.iter, self.batch in enumerate(self.dl):
            self.callback('before_batch')
            self._one_batch(train)
            self.callback('after_batch')

    def fit(self, num_epochs):
        self.num_epochs = num_epochs
        
        self.callback('before_fit')

        for self.epoch in range(self.num_epochs):
            self.callback('before_epoch')
            self._one_epoch(True)
            self.callback('after_epoch')

        self.callback('after_fit')
    
    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)