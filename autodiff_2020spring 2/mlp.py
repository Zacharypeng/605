"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

EPS=1e-4

def fwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.eval(network.my_xman.operationSequence(network.my_xman.loss), valueDict)

def bwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.bprop(network.my_xman.operationSequence(network.my_xman.loss), valueDict,loss=np.float_(1.0))

def update(network, dataParamDict, grads, rate):
    for rname in grads:
        if network.my_xman.isParam(rname):
            dataParamDict[rname] = dataParamDict[rname] - rate*grads[rname]
    return dataParamDict

def accuracy(probs, targets):
    preds = np.argmax(probs, axis=1)
    targ = np.argmax(targets, axis=1)
    return float((preds==targ).sum())/preds.shape[0]

def grad_check(network):
    # function which takes a network object and checks gradients
    # based on default values of data and params
    dataParamDict = network.my_xman.inputDict()
    fd = fwd(network, dataParamDict)
    grads = bwd(network, fd)
    for rname in grads:
        if network.my_xman.isParam(rname):
            fd[rname].ravel()[0] += EPS
            fp = fwd(network, fd)
            a = fp['loss']
            fd[rname].ravel()[0] -= 2*EPS
            fm = fwd(network, fd)
            b = fm['loss']
            fd[rname].ravel()[0] += EPS
            auto = grads[rname].ravel()[0]
            num = (a-b)/(2*EPS)
            if not np.isclose(auto, num, atol=1e-3):
                raise ValueError("gradients not close for %s, Auto %.5f Num %.5f"
                        % (rname, auto, num))

def glorot(m,n):
    # return scale for glorot initialization
    return np.sqrt(6./(m+n))

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)-1
        self.my_xman = self._build(layer_sizes) # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable
        print(self.my_xman.operationSequence(self.my_xman.loss))

    def _build(self, layer_sizes):
        print("INITIAZLIZING with layer_sizes:", layer_sizes)
        self.params = {}
        for i in range(self.num_layers):
            k = i+1
            sc = glorot(layer_sizes[i], layer_sizes[i+1])
            self.params['W'+str(k)] = f.param(name='W'+str(k),
                    default=sc*np.random.uniform(low=-1.,high=1.,
                        size=(layer_sizes[i], layer_sizes[i+1])))
            self.params['b'+str(k)] = f.param(name='b'+str(k),
                    default=0.1*np.random.uniform(low=-1.,high=1.,size=(layer_sizes[i+1],)))
        self.inputs = {}
        self.inputs['X'] = f.input(name='X', default=np.random.rand(1,layer_sizes[0]))
        self.inputs['y'] = f.input(name='y', default=np.random.rand(1,layer_sizes[-1]))
        x = XMan()
        inp = self.inputs['X']
        for i in range(self.num_layers):
            oo = f.mul(inp,self.params['W'+str(i+1)]) + self.params['b'+str(i+1)]
            inp = f.relu( oo )

        x.output = f.softMax(inp)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        return x.setup()

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len,
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len,
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len,
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print("building mlp...")
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    grad_check(mlp)

    print("done")

    # train
    print("training...")
    logger = open('%s_mlp4c_L%d_H%d_B%d_E%d_lr%.3f.txt'%
            (dataset,max_len,num_hid,batch_size,epochs,init_lr),'w')
    # get default data and params
    value_dict = mlp.my_xman.inputDict()
    min_loss = 1e5
    lr = init_lr
    train_loss = np.ndarray([0])
    best_param_dict = {}
    for i in range(epochs):
        for ii, (idxs,e,l) in enumerate(mb_train):
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.items():
                value_dict[k] = v
            # fwd-bwd
            vd = fwd(mlp,value_dict)
            gd = bwd(mlp,value_dict)
            value_dict = update(mlp, value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')
            train_loss = np.append(train_loss, vd['loss'])
            # print(ii)
        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (idxs,e,l) in mb_valid:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.items():
                value_dict[k] = v
            # fwd
            vd = fwd(mlp, value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += 1
        acc = accuracy(np.vstack(probs), np.vstack(targets))
        c_loss = tot_loss/n
        if c_loss<min_loss:
            min_loss = c_loss
            for k,v in value_dict.items():
                best_param_dict[k] = np.copy(v)
        message = ('Epoch %d VAL loss %.3f min_loss %.3f acc %.3f' %
                (i,c_loss,min_loss,acc))
        logger.write(message+'\n')
        print(message)

    np.save(train_loss_file, train_loss)

    tot_loss, n = 0., 0
    probs = []
    targets = []
    for (idxs,e,l) in mb_test:
        # prepare input
        data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
        for k,v in data_dict.items():
            best_param_dict[k] = v
        # fwd
        vd = fwd(mlp,best_param_dict)
        tot_loss += vd['loss']
        probs.append(vd['output'])
        targets.append(l)
        n += 1
    acc = accuracy(np.vstack(probs), np.vstack(targets))
    c_loss = tot_loss/n
    np.save(output_file, np.vstack(probs))
    print("done, test loss = %.3f acc = %.3f" % (c_loss, acc))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='../../data/tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
