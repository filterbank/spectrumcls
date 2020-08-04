import tensorflow as tf
import numpy as np

global classnumber,tpr, fpr, tpfp


def tpr_fpr_init(num_class):
    global tpr, fpr, tpfp, classnumber
    classnumber = num_class
    tpr = np.zeros(classnumber, dtype=np.float32)
    fpr = np.zeros(classnumber, dtype=np.float32)
    tpfp = np.zeros((classnumber, classnumber), dtype=np.uint16)
    return tpr,fpr,tpfp,classnumber

def tpr_fpr_argmax(pred,label):
    argmaxlist = []
    predvalue = tf.argmax(pred,1)
    labelvalue = tf.argmax(label,1)
    argmaxlist.append(predvalue)
    argmaxlist.append(labelvalue)
    return argmaxlist

def tpr_fpr_statistics(argmaxlist):
    global tpr, fpr, tpfp
    predvalue = argmaxlist[0]
    labelvalue = argmaxlist[1]
    for (i, j) in zip(labelvalue, predvalue):
        tpfp[i][j] += 1
    return tpfp
	
def tpr_fpr_compute():
    global tpr, fpr, tpfp,classnumber
    for k in np.arange(classnumber):
        tpr[k] = (tpfp[k][k]*1.0)/(tpfp[k].sum())
        fpr[k] = ((tpfp[:,k].sum()-tpfp[k][k])*1.0)/(tpfp.sum()-tpfp[k].sum())
    return tpr,fpr

'''
def tpr_fpr(pred,label,num_class,init=True):
    if init:
        tpr_fpr_init(num_class)
	tpr_fpr_statistics(pred,label)
    return tpr_fpr_compute()


if __name__ == '__main__':
    global tpr, fpr, tpfp, classnumber
    pred = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0],
                     [0, 1, 0], [0, 1, 0], [0, 0, 1],
                     [0, 0, 1 ],[0, 0, 1],[1, 0, 0],[0, 1, 0]])
    label = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0],
                      [0, 1, 0], [0, 1, 0], [0, 1, 0],
                      [0, 0, 1 ],[0, 0, 1],[0, 0, 1],[0, 0, 1]])
    classnumber = 3
    tpr_fpr_init(classnumber)
    argmax = tpr_fpr_argmax(pred,label)
    x = tf.placeholder("float", [10, 3])
    y = tf.placeholder("float", [10, 3])
    sess = tf.Session()
    argmaxlist = sess.run(argmax, feed_dict={x:pred,y:label})
    tpr_fpr_statistics(argmaxlist)
    tpr_fpr_statistics(argmaxlist)
    tpr_fpr_compute()

    print tpfp
    for i in np.arange(classnumber):
        print ('TPR%s = %s,FPR%s = %s' % (i, tpr[i], i, fpr[i]))
'''
'''
    tpr_fpr(pred, label, init=True)
    print tpfp
    for i in np.arange(classnumber):
        print ('TPR%s = %s,FPR%s = %s' % (i,tpr[i],i,fpr[i]))
'''