import polars as pl
import matplotlib.pyplot as plt
import numpy as np

points = 100

powMax = 80000

def torque_power(rpm, gearRatio):
    motorHz = rpm*2*np.pi/60
    motorTorque = min(powMax/motorHz, 180)
    power = motorTorque*motorHz
    wheelTorque = motorTorque*gearRatio
    return wheelTorque, power/1000

rpms = np.arange(0, 7500, 7500/points)
gearRatios = np.arange(3.5, 4.5, 1/points)

R, G = np.meshgrid(rpms, gearRatios)
print(R.shape)
print(G.shape)

P = np.zeros_like(R)
T = np.zeros_like(R)

for i in range(points):
    for j in range(points):
        T[i,j], P[i,j] = torque_power(R[i,j], G[i,j])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')

print(np.shape(T))
print(np.shape(P))

ax1.scatter(np.reshape(R,(points**2,1)), np.reshape(G,(points**2,1)), 4*np.reshape(P,(points**2,1)), s=0.3)
ax1.scatter(np.reshape(R,(points**2,1)), np.reshape(G,(points**2,1)), np.reshape(T,(points**2,1)), s=0.3)
# ax2.plot(R, G, T)

plt.show()

