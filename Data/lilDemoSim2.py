## Double pendulum simulation
## Nathaniel Platt 6/23-26/25

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np
from scipy.integrate import RK45

@dataclass
class State:
    angle: float
    omega: float                                                                                                                                               
    radius: float
    mass: float
    rod: bool


## Earlier attempts at solving this with kinematics proved incredibly difficult beccause the tension between the two nodes is beyond my ability to calculate effectively
# I started with a cartesian model but thought a polar model might be easier (thinking of everything in angles and torques instead)
# It did seem to make it easier but the tension got me in the end.
# I chose a coordinate system where up is 0 deg, CW is positive and CCW is negative. This made thinking about it a lot easier, but again, tension got me
# Energy still doesn't make sense. The number in the console is energy (Joules) combined PE and KE which should be constant so I'm doing my math wrong somewhere. 
# Couldn't figure out how to calculate energy of the second far out node. Is it just its own angular acceleration? (no lol)
def update(s1: State, s2: State):
    global dt
    # g2 = s2.mass * 9.809
    # g1 = s1.mass * 9.809
    # # t2 = g2 * np.cos(s2.angle) * (-1 if abs(s2.angle) < 90 else 1) * np.sign(s2.angle)
    # t2 = s2.mass * (s2.omega**2) * s2.radius + np.cos(s2.angle) * g2
    # f2 = g2 * np.sin(s2.angle) #+ t2 * np.sin((np.pi / 2) - abs(s2.angle - s1.angle))
    # f1 = g1 * np.sin(s1.angle) + t2 * np.sin(s2.angle - s1.angle)
    # i1 = s1.radius**2 * s1.mass if not s1.rod else s1.radius**2 * s1.mass / 3
    # i2 = s2.radius**2 * s2.mass if not s2.rod else s2.radius**2 * s2.mass / 3
    # torque1 = f1 * s1.radius
    # alpha1 = torque1 / i1
    # torque2 = f2 * s2.radius
    # alpha2 = torque2 / i2
    # alpha1 = (-9.81*(2*s1.mass + s2.mass)*np.sin(s1.angle) - s2.mass*9.81*np.sin(s1.angle - 2*s2.angle) - 2*np.sin(s1.angle - s2.angle)*s2.mass*((s2.omega**2)*s2.radius + (s1.omega**2)*s1.radius*np.cos(s1.angle-s2.angle)))/(s1.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*s1.angle-2*s2.angle)))
    # alpha2 = (2*np.sin(s1.angle-s2.angle)*((s1.omega**2)*s1.radius*(s1.mass+s2.mass)+9.81*(s1.mass+s2.mass)*np.cos(s1.angle)+(s2.omega**2)*s2.radius*s2.mass*np.cos(s1.angle-s2.angle)))/(s2.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*s1.angle-2*s2.angle)))
    # # alpha1 = (-9.81*(2*s1.mass + s2.mass) * np.sin(s1.angle) - s2.mass*9.81*np.sin(s1.angle + 2*s2.angle) - 2*np.sin(s1.angle + s2.angle)*s2.mass*((s2.omega**2)*s2.radius + (s1.omega**2)*s1.radius*np.cos(s1.angle + s2.angle)))/(s1.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*s1.angle+2*s2.angle)))
    # # alpha2 = (2*np.sin(s1.angle + s2.angle) * ((s1.omega**2) * s1.radius*(s1.mass+s2.mass)+9.81*(s1.mass+s2.mass)*np.cos(s1.angle)+(s2.omega**2)*s2.radius*s2.mass*np.cos(s1.angle+s2.angle)))/(s2.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*s1.angle+2*s2.angle)))
    # s1.omega += alpha1 * dt
    # s2.omega += alpha2 * dt
    # s1.angle += s1.omega * dt
    # s2.angle += s2.omega * dt
    # while s1.angle > np.pi:
    #     s1.angle -= np.pi * 2
    # while s1.angle < -1 * np.pi:
    #     s1.angle += np.pi * 2
    # while s2.angle > np.pi:
    #     s2.angle -= np.pi * 2
    # while s2.angle < -1 * np.pi:
    #     s2.angle += np.pi * 2
    # s1y = np.cos(s1.angle)*s1.radius
    # s2y = np.cos(s2.angle)*s2.radius + s1y
    # p1 = s1y * 9.809 * s1.mass
    # p2 = s2y * 9.809 * s2.mass
    # k1 = 0.5 * i1 * s1.omega**2
    # v2 = np.sqrt(s1.radius**2 * s1.omega**2 + s2.radius**2 * s2.omega**2 + 2*s1.radius*s2.radius*s1.omega*s2.omega*np.cos(s1.angle-s2.angle))
    # k2 = 0.5 * s2.mass * v2**2
    # Kinetic Energy
    T1 = 0.5 * s1.mass * (s1.omega * s1.radius)**2
    T2 = 0.5 * s2.mass * ((s1.omega * s1.radius * np.cos(s1.angle)) + (s2.omega * s2.radius * np.cos(s2.angle)))**2

    # Potential Energy
    U1 = s1.mass * 9.81 * (s1.radius * (1 - np.cos(s1.angle)))
    U2 = s2.mass * 9.81 * (s1.radius * (1 - np.cos(s1.angle)) + s2.radius * (1 - np.cos(s2.angle)))

    # Total Energy
    energy = T1 + T2 + U1 + U2
    # energy = p1 + p2 + k1 + k2
    # print(f"{s1}, {s2}, \nF1: {g1 * np.sin(s1.angle)} + {t2 * np.sin(s2.angle - s1.angle)}, \nT2: {t2}")
    print(energy)
    return s1, s2

## Step accuracy really mattered in this simulation. With a riemann sum, it lost energy very quickly and mostly came to a rest for some reason.

def rateRK(t, y):
    global states
    s1 = states[0]
    s2 = states[1]
    theta1 = y[0]
    theta2 = y[1]
    omega1 = y[2]
    omega2 = y[3]
    alpha1 = (-9.81*(2*s1.mass + s2.mass)*np.sin(theta1) - s2.mass*9.81*np.sin(theta1 - 2*theta2) - 2*np.sin(theta1 - theta2)*s2.mass*((omega2**2)*s2.radius + (omega1**2)*s1.radius*np.cos(theta1-theta2)))/(s1.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*theta1-2*theta2)))
    alpha2 = (2*np.sin(theta1-theta2)*((omega1**2)*s1.radius*(s1.mass+s2.mass)+9.81*(s1.mass+s2.mass)*np.cos(theta1)+(omega2**2)*s2.radius*s2.mass*np.cos(theta1-theta2)))/(s2.radius*(2*s1.mass+s2.mass-s2.mass*np.cos(2*theta1-2*theta2)))
    return [omega1, omega2, alpha1, alpha2]

def animate(i):
    global solver
    # plt.pause(0.1)
    ax.clear()
    s1 = states[0]
    s2 = states[1]

    solver.step()
    y = solver.y

    s1.angle = y[0]
    s2.angle = y[1]
    s1.omega = y[2]
    s2.omega = y[3]

    s1x = np.sin(s1.angle)*s1.radius
    s1y = -1 * np.cos(s1.angle)*s1.radius
    s2x = np.sin(s2.angle)*s2.radius + s1x
    s2y = -1 * np.cos(s2.angle)*s2.radius + s1y
    # s1x = np.cos(s1.angle)*s1.radius
    # s1y = np.sin(s1.angle)*s1.radius
    # s2x = np.cos(s2.angle)*s2.radius + s1x
    # s2y = np.sin(s2.angle)*s2.radius + s1y
    ax.plot([0, s1x, s2x], [0, s1y, s2y])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks([])
    ax.set_yticks([])   
    states[0:] = update(s1, s2)
    # print(s0)
    return ax, 



dt = 0.025

s1 = State(angle = np.pi + 0.11, omega = 0, radius = 1, mass = 1, rod = False)
s2 = State(angle = np.pi + 0.1, omega = 0, radius = 1, mass = 1, rod = False)
print(s1)
print(s2)
# print(f"m:{m(s0.x1, s0.x2)}")
states = [s1, s2]
solver = RK45(rateRK, 0, [s1.angle, s2.angle, s1.omega, s2.omega], t_bound=np.inf, max_step=0.025)
fig = plt.figure(figsize=(3,3), dpi=150)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("")
ax.set_ylabel("")
ax.plot([],[])
plt.pause(5)
# plt.pause(10)
ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()


# def normalize(s: State):
#     m1 = m(s.x1, s.y1)
#     s.x1 = s.x1/m1*s.l1
#     s.y1 = s.y1/m1*s.l1
#     m2 = m(s.x2 - s.x1, s.y2 - s.y1)
#     s.x2 = s.x1 + (s.x2 - s.x1)/m2*s.l2
#     s.y2 = s.y1 + (s.y2 - s.y1)/m2*s.l2
#     return s