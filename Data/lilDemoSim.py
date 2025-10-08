## Basic Ball Bouncing simulation
## Nathaniel Platt Early June 2025

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    color: float


def update_pos(state: State):
    state.x += state.vx * dt
    state.y += state.vy * dt
    state.vy -= 9.81 * dt  # Gravity effect
    if state.y < 0:
        # state.vy = -state.vy * 1.01
        state.y = 0
        state.vy = -state.vy * 0.7
    # if state.x < 0 or state.x > 100:
    #     state.vx = -state.vx * 1.01

    return state

def animate(i):
    global slist
    global line
    slist = [update_pos(s) for s in slist]
    ax.clear()
    ax.scatter([s.x for s in slist], [s.y for s in slist],c=[s.color for s in slist], s=0.1, cmap='viridis')
    ax.set_xlim(-20, 700)
    ax.set_ylim(0, 100)
    return ax, 

dt = 0.1
# s0 = State(x=0.0, y=20.0, vx=1.0, vy=5.0)
slist = []
for i in range(0, 50, 15):
    for j in range(0, 50, 15):
        for k in range(5):
            for l in range(5):
                slist.append(State(x=i+np.random.rand()*20, y=j+np.random.rand()*20, vx=k+np.random.rand()*2, vy=l+np.random.rand()*2, color=j))

fig = plt.figure(figsize=(3,3), dpi=100)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-20, 700)
ax.set_ylim(0, 100)
ax.scatter([],[], s=0.1, c='blue')
# plt.pause(10)

ani = animation.FuncAnimation(fig, animate, interval=0, blit=False)
plt.show()
