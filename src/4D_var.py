import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.optimize import minimize

Nt = 520
dt = 0.01
t = np.arange(0, Nt*dt, dt)
t_obs = np.arange(49, Nt, 50)

x0_T = 2
k0_T = 1
z0_T = [x0_T, k0_T]

def simulation(z0):
    z = np.zeros((2, Nt))
    z[:, 0] = z0
    
    for i in range(Nt-1):
        z[0][i+1] = z[0][i] - z[1][i] * z[0][i]**2 * dt
        z[1][i+1] = z[1][i]
    
    return z


def get_J(z0):
    z = simulation(z0)
    res = 0
    for i in range(len(t_obs)):
        res += 0.5 * (z[0][t_obs[i]] - y[t_obs[i]])**2
    return res


def gradJ(z0):
    z = simulation(z0)
    lmd = np.zeros((2, Nt))
    for i in range(Nt-2, -1, -1):
        lmd[0][i] = lmd[0][i+1] + 2 * z[1][i] * z[0][i] * lmd[0][i+1] * dt
        lmd[1][i] = lmd[1][i+1] + z[0][i] * z[0][i] * lmd[0][i+1] * dt
        
        if i in t_obs:
            lmd[0][i] += (z[0][i] - y[i]) * dt
    return lmd


# true value
z_T = simulation(z0_T)

# observed value
y = z_T[0] + np.random.normal(0, 0.05, len(z_T[0]))

# 推定の初期値
x0_init = 3
k0_init = 3
z0_init = np.array([x0_init, k0_init])
z_init = simulation(z0_init)

# 最適化
opt = minimize(get_J, z0_init, method='CG', options={'maxiter':500000})
z0_pred = opt.x

# シミュレーション
z_pred = simulation(z0_pred)

# プロット
fig = plt.figure(figsize=(10, 8))
plt.plot(t, z_T[0], c='black', label='True value')
plt.plot(t, z_pred[0], c='r', label='Predicted value using optimized initial value')
plt.plot(t, z_init[0], c='b', label='Predicted value using initial value before optimization')
plt.scatter((t_obs+1)*dt, y[t_obs], facecolor='None', edgecolor='black', label='Observed value')
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig('../output/result.png', dpi=200, bbox_inches='tight')

print(z0_pred)