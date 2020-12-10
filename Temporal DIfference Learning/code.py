import numpy as np
import random
import matplotlib.pyplot as plt

truth=[1/6.0, 1/3.0, 1/2.0, 2/3.0, 5/6.0,]

def randomWalk():
    state=3
    vec=[0]*5
    vec[2]=1
    seq=[vec]
    while state not in (0,6):
        a = random.choice([-1, 1])
        state+=a
        if state==6:
            seq.append(1)
        elif state==0:
            seq.append(0)
        else:
            vec = [0] * 5
            vec[state-1] = 1
            seq.append(vec)
    return seq   

def construct(n_sets=100, n_sequences=10):
    dsets=[]
    for i in range(n_sets):
        sequences=[]
        for j in range(n_sequences):
            seq=randomWalk()
            sequences.append(seq)
        dsets.append(sequences)
    return dsets

def RMS(a,b):
    return np.sqrt(np.mean(np.square(b-a)))

def experiment1(lam,alp,dsets,init_w=np.array([0.5]*5),eps=0.001):
    rmse=0
    for p in range(100):
        w=init_w
        chg=1
        while chg>eps:
            delta = np.zeros(5, )
            for q in range(10):
                sequences = dsets[p][q]
                for t in range(1,len(sequences)):
                    lam_gradP = np.zeros(5, )
                    for k in range(t):
                        lam_gradP += (lam ** (t - 1 - k)) * np.array(sequences[k])
                    if t < len(sequences)-1:
                        delta += alp * (w.T.dot(sequences[t]) - w.T.dot(sequences[t - 1])) * lam_gradP
                    else:
                        delta += alp * (sequences[t] - w.T.dot(sequences[t - 1])) * lam_gradP
            w+=delta
            chg=max(np.abs(delta))
        rmse+=RMS(truth,w)
    return rmse/100.0

def experiment2(lam,alp,dsets):
    rmse=0
    for p in range(100):
        w = np.array([0.5] * 5)
        for q in range(10):
            delta = np.zeros(5, )
            sequences = dsets[p][q]
            for t in range(1,len(sequences)):
                lam_gradP = np.zeros(5, )
                for k in range(t):
                    lam_gradP += (lam ** (t - 1 - k)) * np.array(sequences[k])
                if t < len(sequences)-1:
                    delta += alp * (w.T.dot(sequences[t]) - w.T.dot(sequences[t - 1])) * lam_gradP
                else:
                    delta += alp * (sequences[t] - w.T.dot(sequences[t - 1])) * lam_gradP
            w+=delta
        rmse+=RMS(truth,w)
    return rmse/100.0

random.seed(111)
dsets=construct()

lam_f3=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
err_f3=[]
for l in lam_f3:
    err=experiment1(l,0.01,dsets)
    err_f3.append(err)

plt.clf()
plt.plot(lam_f3, err_f3, marker='o')
plt.text(0.8,0.18,'Widrow-Hoff')
plt.title('Figure 3')
plt.ylabel('ERROR')
plt.xlabel('\u03BB')
plt.savefig('Figure3.png')

lam_f4=[0, 0.3, 0.8, 1]
alp_f4=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]

err_f4={}
for l in lam_f4:
    lst=[]
    for a in alp_f4:
        err = experiment2(l, a, dsets)
        if err<1:
            lst.append(err)
    err_f4[str(l)]=lst

plt.clf()
plt.plot(alp_f4[:len(err_f4['0'])], err_f4['0'], marker='o', label='\u03BB = 0')
plt.plot(alp_f4[:len(err_f4['0.3'])], err_f4['0.3'], marker='o', label='\u03BB = .3')
plt.plot(alp_f4[:len(err_f4['0.8'])], err_f4['0.8'], marker='o', label='\u03BB = .8')
plt.plot(alp_f4[:len(err_f4['1'])], err_f4['1'], marker='o', label='\u03BB = 1')
plt.text(0.29,0.88,'Widrow-Hoff')
plt.title('Figure 4')
plt.ylabel('ERROR')
plt.xlabel('\u03B1')
plt.yticks(np.arange(0.1,1,0.1))
plt.legend()
plt.savefig('Figure4.png')

lam_f5=np.arange(0,1.1,0.1)
alp_f5=np.arange(0.01,0.51,0.01)

err_f5=[]
for l in lam_f5:
    lst=[]
    for a in alp_f5:
        err = experiment2(l, a, dsets)
        lst.append(err)
    err_f5.append(min(lst))

plt.clf()
plt.plot(lam_f5, err_f5, marker='o')
plt.ylabel('ERROR USING BEST \u03BB')
plt.text(0.8,0.185,'Widrow-Hoff')
plt.title('Figure 5')
plt.xlabel('\u03BB')
plt.savefig('Figure5.png')