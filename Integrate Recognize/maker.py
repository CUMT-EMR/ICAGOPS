import numpy as np
from scipy.optimize import minimize
import itertools
import pandas as pd
Theta = ['risk', 'norisk']
power_set = [frozenset(s) for i in range(len(Theta) + 1)
             for s in itertools.combinations(Theta, i)]

def bpa_from_probability(x):

    m = {A: 0.0 for A in power_set}
    m[frozenset(['risk'])] = x
    m[frozenset(['norisk'])] = 1 - x
    return m

def compute_alpha(m_i, m_j):
    alpha = {}
    for A in power_set:
        for B in power_set:
            pij = m_i[A] * m_j[B]
            denom = max(m_i[A] * m_j[B], 1e-9)
            alpha[(A, B)] = pij / denom
    return alpha

def maker_fuse(mA, mB, mC, rA, rB, rC, alphaAB, alphaAC, alphaBC):
    fused = {C: 0.0 for C in power_set}
    conflict = 0.0

    for A in power_set:
        for B in power_set:
            for C in power_set:
                wA = rA[A] * mA[A]
                wB = rB[B] * mB[B]
                wC = rC[C] * mC[C]
                gamma = alphaAB[(A,B)] * alphaAC[(A,C)] * alphaBC[(B,C)]
                contrib = wA * wB * wC * gamma
                if A & B & C:
                    fused[A & B & C] += contrib
                else:
                    conflict += contrib

    total = sum(fused.values())
    if total <= 0:
        total = 1.0
    for C in fused:
        fused[C] /= total
    return fused

def train_maker(X, Y):

    S = len(X)
    r0 = [{A: 0.8 for A in power_set} for _ in range(3)]

    def pack(r):
        return np.array([r[i][A] for i in range(3) for A in power_set])
    def unpack(theta):
        r = []
        idx = 0
        for _ in range(3):
            r_dict = {}
            for A in power_set:
                r_dict[A] = theta[idx]
                idx += 1
            r.append(r_dict)
        return r
    def loss_fn(theta):
        r = unpack(theta)
        loss = 0.0
        for s in range(S):
            mA = bpa_from_probability(X[s][0])
            mB = bpa_from_probability(X[s][1])
            mC = bpa_from_probability(X[s][2])
            alphaAB = compute_alpha(mA, mB)
            alphaAC = compute_alpha(mA, mC)
            alphaBC = compute_alpha(mB, mC)
            fused = maker_fuse(mA, mB, mC, r[0], r[1], r[2], alphaAB, alphaAC, alphaBC)
            loss += (1 - fused[frozenset([Y[s]])])**2
        return loss / S

    theta0 = pack(r0)
    bounds = [(1e-3, 1.0)] * len(theta0)
    res = minimize(loss_fn, theta0, bounds=bounds, method='L-BFGS-B')
    return unpack(res.x)

if __name__ == "__main__":

    df_X_hist = pd.read_excel('proboutput.xlsx',usecols=[1,2,3])[:790]
    df_Y_hist = pd.read_excel('proboutput.xlsx',usecols=[4])[:790]
    X_hist = df_X_hist.values.tolist()
    Y_hist = df_Y_hist.iloc[:, 0].tolist()
    r_opt = train_maker(X_hist, Y_hist)
    df_X_hist = pd.read_excel('probvalidateoutput.xlsx',usecols=[1,2,3])[:690]
    df_Y_hist = pd.read_excel('probvalidateoutput.xlsx',usecols=[4])[:690]
    X_hist = df_X_hist.values.tolist()
    Y_hist = df_Y_hist.iloc[:, 0].tolist()
    r_opt = train_maker(X_hist, Y_hist)
    x1, x2, x3 = 0.45, 0.85, 0.85
    df_X_hist = pd.read_excel('probvalidateoutput.xlsx',usecols=[1,2,3])
    X_hist = df_X_hist.values.tolist()
    weixian_list = []
    notweixian_list = []
    notsure = []
    for item in X_hist:
        x1, x2, x3 = item[0], item[1], item[2]
        mA = bpa_from_probability(x1)
        mB = bpa_from_probability(x2)
        mC = bpa_from_probability(x3)
        alphaAB = compute_alpha(mA, mB)
        alphaAC = compute_alpha(mA, mC)
        alphaBC = compute_alpha(mB, mC)
        fused = maker_fuse(mA, mB, mC, r_opt[0], r_opt[1], r_opt[2], alphaAB, alphaAC, alphaBC)
        idx =0
        for subset, mass in fused.items():
            if subset:
                if idx==0:
                    weixian_list.append(mass)
                if idx ==1:
                    notweixian_list.append(mass)
                if idx == 2:
                    notsure.append(mass)
                print(f"{set(subset)}: {mass:.4f}")
                idx = idx+1

    decision = max((s for s in fused if len(s)==1), key=lambda s: fused[s])
    print("eventual recognition：", list(decision)[0])

df_X_hist["risk"] = weixian_list
df_X_hist["norisk"] = notweixian_list
df_X_hist["notsure"] = notsure
df_X_hist.to_excel("./resultexport.xlsx")
