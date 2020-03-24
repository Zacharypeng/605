#!/usr/bin/env python

import mlp
from functions import *
from utils import *
import argparse
from autograd import Autograd
import time
import os

import contextlib
import sys
import cStringIO

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    saved_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    try:
        yield
    except Exception:
        saved_output = sys.stdout
        sys.stdout = saved_stdout
        print saved_output.getvalue()
        raise
    sys.stdout = saved_stdout

EPS = 1e-4

LOSS_INV_MAX = 20 

MLP_LOSS_THRESHOLD = [(1.022,15), (1.5,10), (2.0, 0)]
MLP_TIME_THRESHOLD = [(25,15), (50,10), (100,0)]
MLP_LOSS_INV_THRESHOLD = [(LOSS_INV_MAX/2,10), (LOSS_INV_MAX*3/4,5), (LOSS_INV_MAX,0)]

#Loss : 15
#Loss order : 10
#Time : 15
#Gradient : 20
#Function : 10 


def linear(thresholds, x):
    return float(thresholds[1][1]-thresholds[0][1])*(x- thresholds[0][0])/(thresholds[1][0]-thresholds[0][0])+thresholds[0][1]
""" x1 and x2 are two tuples """
def linear_mark(thresholds, x):
    if x<=thresholds[0][0]:
        return thresholds[0][1]
    elif x<=thresholds[1][0]:
        return linear(thresholds[:2], x)
    elif x<=thresholds[2][0]:
        return linear(thresholds[1:3], x)
    else:
        return thresholds[2][1]

def _crossEnt(x,y):
    log_x = np.nan_to_num(np.log(x))
    return - np.multiply(y,log_x).sum(axis=1, keepdims=True)

def fwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.eval(network.my_xman.operationSequence(network.my_xman.loss), valueDict)

def bwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.bprop(network.my_xman.operationSequence(network.my_xman.loss), valueDict,loss=np.float_(1.0))
def load_params_from_file(filename):
    return np.load(filename)[()]

def save_params_to_file(d, filename):
    np.save(filename, d)

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


def loss_inv_check(loss_arr):
    result["mlp_train_loss"] = 0
    try:
        in_arr = np.reshape(loss_arr, [len(loss_arr)])
        if len(in_arr) < LOSS_INV_MAX:
            raise ValueError("Not enough Train Loss measurements. Found only %d train loss entries!"%len(loss_arr))
        in_arr = in_arr[0:LOSS_INV_MAX]
        inv = 0
        for i in range(LOSS_INV_MAX-1):
            if in_arr[i] < in_arr[i+1]:
                inv += 1
        
        result["mlp_train_loss"] =  linear_mark(MLP_LOSS_INV_THRESHOLD, inv)
        return inv
    except Exception, e:
        print "MLP TRAIN LOSS FAILED" 
        print e    
        return -1

def test_allclose(test_name, real, expected, rtol=1e-7, score=1):
    try:
        np.testing.assert_allclose(real, expected, rtol=rtol)
        result["mlp_unit_tests"] += score
    except Exception, e:
        print "MLP UNIT TESTS %s FAILED" % test_name
        print e

def mlp_unit_tests():
    x = np.array([
        [ 0.76677119,  0.12815245],
        [ 0.4007303 ,  0.77046941],
        [ 0.00574018,  0.71242641]])
    y = np.array([
        [-0.06655641,  0.10877971],
        [ 0.13663944, -0.12461873]])
    z = np.array([[0., 1.], [0., 1.], [1., 0.]])
    v =np.array([[ 0.96894013], [ 0.07382228]])
    # Eval mul
    expected_x_mul_y =  np.array([[-0.03352286,  0.06743895],
        [ 0.07860534, -0.05242359],
        [ 0.0969635 , -0.08815726]])
    test_allclose('Eval mul', EVAL_FUNS['mul'](x, y), expected_x_mul_y)
    expected_relu_y = np.array([
        [ 0.        ,  0.10877971],
        [ 0.13663944,  0.        ]])
    # Eval relu
    test_allclose('Eval relu', EVAL_FUNS['relu'](y), expected_relu_y)
    expected_softMax_x = np.array([
        [ 0.65444116,  0.34555884],
        [ 0.40860406,  0.59139594],
        [ 0.33033148,  0.66966852]])
    # Eval softMax
    test_allclose('Eval softMax', EVAL_FUNS['softMax'](x), expected_softMax_x)
    expected_crossEnt_softMax_x_z = np.array([
        [ 1.06259235],
        [ 0.52526954],
        [ 1.10765864]])
    # Eval crossEnt
    test_allclose('Eval crossEnt', EVAL_FUNS['crossEnt'](expected_softMax_x, z), expected_crossEnt_softMax_x_z)
    # Eval mean
    expected_mean_v = 0.52138120499999996
    test_allclose('Eval mean', EVAL_FUNS['mean'](v), expected_mean_v)
    # BP mul
    delta_x_mul_y = np.array([
        [ 0.12523631,  0.00680066],
        [ 0.48109275,  0.95663136],
        [ 0.40436419,  0.56481742]])
    test_allclose('BP mul 0', BP_FUNS['mul'][0](delta_x_mul_y, expected_x_mul_y, x, y), np.array([
        [-0.00759551,  0.01626473],
        [ 0.07204228, -0.05347794],
        [ 0.03452765, -0.01513473]]), rtol=1e-06)
    test_allclose('BP mul 1', BP_FUNS['mul'][1](delta_x_mul_y, expected_x_mul_y, x, y), np.array([
        [ 0.29113716,  0.39180788],
        [ 0.67479632,  1.14031757]]))
    # BP relu
    delta_relu_y = np.array([
        [ 0.66202207,  0.59765468],
        [ 0.01812402,  0.58537534]])
    test_allclose('BP relu', BP_FUNS['relu'][0](delta_relu_y, expected_relu_y, y), np.array([
        [ 0.        ,  0.59765468],
        [ 0.01812402,  0.        ]]))
    # BP crossEnt-softMax
    delta_crossEnt_softMax_x_z = np.array([
        [  5.69906247e-01],
        [  8.66851385e-01],
        [  2.79581480e-04]])
    test_allclose('BP crossEnt-softMax', BP_FUNS['crossEnt-softMax'][0](delta_crossEnt_softMax_x_z, expected_crossEnt_softMax_x_z, x, z), np.array([
        [  3.72970104e-01,  -3.72970104e-01],
        [  3.54198998e-01,  -3.54198998e-01],
        [ -1.87226917e-04,   1.87226917e-04]]))
    # BP mean
    test_allclose('BP mean', BP_FUNS['mean'][0](0.19950823, expected_mean_v, v), np.array([
        [ 0.09975412],
        [ 0.09975412]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='../data/autolab')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1)
    parser.add_argument('--mlp_init_lr', dest='mlp_init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    parser.add_argument('--mlp-solution', dest='result_mlp_file', type=str, default='MLP_solution')

    params = vars(parser.parse_args())
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    output_file = params['output_file']
    mlp_file = params['result_mlp_file']
    train_loss = params['train_loss_file']

    # load data and preprocess
    with nostdout():
        dp = DataPreprocessor()
        data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test.solution'%dataset)
        # minibatches
        mb_train = MinibatchLoader(data.training, batch_size, max_len,
               len(data.chardict), len(data.labeldict))
        mb_test = MinibatchLoader(data.test, len(data.test), max_len,
               len(data.chardict), len(data.labeldict), shuffle=False)

        result = {}
        result["mlp_grad_check"] = 0
        result["mlp_accuracy"] = 0
    
        result["mlp_time"] = 0
        targets = []
        indices = []
        validation = 1000
        for (idxs,e,l) in mb_test:
            targets.append(l)
            indices.extend(idxs)
        try:
            result["mlp_unit_tests"] = 0
            try:
                mlp_unit_tests()
    #         except Exception, e:
    #             print "MLP UNIT TESTS FAILED"
    #             print e
            except ValueError:
                pass
            
            try:
                # build
                print "building mlp..."
                Mlp = mlp.MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
                print "checking gradients..."
                grad_check(Mlp)
                result["mlp_grad_check"] = 20
    #         except Exception, e:
    #             print "MLP GRADIENT CHECK FAILED"
    #             print e
    #             result["mlp_grad_check"] = 0
            except ValueError:
                pass
    
            t_start = time.clock()
            # print os.times()
            t_start1 = os.times()[0]
            params["init_lr"] = params["mlp_init_lr"]
            params["output_file"] = output_file+"_mlp"
    
            mlp.main(params)
            mlp_time = time.clock()-t_start
            user_time = os.times()[0]-t_start1
            # print os.times()
    
            student_mlp_loss = _crossEnt(np.load(params["output_file"]+".npy"), np.vstack(targets)).mean()
            # ideal_mlp_loss = _crossEnt(np.load(mlp_file+".npy"), np.vstack(targets)).mean()
    
            print "student_mlp_loss:", student_mlp_loss
            print "mlp_time:", mlp_time, "user_time:", user_time
    
            # print ideal_mlp_loss/student_mlp_loss*10
            result["mlp_accuracy"] =  linear_mark(MLP_LOSS_THRESHOLD, student_mlp_loss)
    
            result["mlp_time"] = linear_mark(MLP_TIME_THRESHOLD, mlp_time)
            
            student_mlp_train_loss_inv = loss_inv_check(np.load(params["train_loss_file"]+".npy"))
            print "student_mlp_train_loss_inv:", student_mlp_train_loss_inv
                                                    
    #     except Exception, e:
    #         print "MLP CHECKING FAILED"
    #         print e
        except ValueError:
            pass

    print "---------------------------------------------------";
    print "Your Autograder's total:", validation, "/ 1000";
    print "---------------------------------------------------";

    print "{ scores: {validation:"+str(validation)+"} }"
    