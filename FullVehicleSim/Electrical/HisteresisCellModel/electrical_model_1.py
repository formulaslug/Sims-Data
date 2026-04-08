import time
import random
import matplotlib.pyplot as plt
from collections import deque

# Store (time, current) values
time_window = 10  # seconds
times = deque()
currents = deque()

plt.ion()  # interactive mode
fig, ax = plt.subplots()

start_time = time.time()

while True:
    now = time.time() - start_time
    current_value = random.uniform(0, 10)  # replace with real sensor current

    # Add new data
    times.append(now)
    currents.append(current_value)

    # Remove old data (>10 seconds)
    while times and times[0] < now - time_window:
        times.popleft()
        currents.popleft()

    # Update plot
    ax.clear()
    ax.plot(times, currents, color='blue')
    ax.set_xlim(max(0, now - time_window), now)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current over last 10 seconds")

    plt.pause(0.1)
