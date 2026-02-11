import serial
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec  # Import gridspec for better layout control
from ahrs.filters import madgwick as mg
from ahrs import Quaternion
from ahrs.common.orientation import q_prod, q_conj



class MadgwickAltered(mg.Madgwick):
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        super().__init__(gyr=gyr, acc=acc, mag=mag, **kwargs)

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt) -> np.ndarray:
        """
        Quaternion Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in nT

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.
        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        if mag is None or not np.linalg.norm(mag)>0:
            return self.updateIMU(q, gyr, acc)
        qDot = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc/a_norm
            m = mag/np.linalg.norm(mag)
            # Rotate normalized magnetometer measurements
            h = q_prod(q, q_prod([0, *m], q_conj(q)))               # (eq. 45)
            bx = np.linalg.norm([h[1], h[2]])                       # (eq. 46)
            bz = h[3]
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Gradient objective function (eq. 31) and Jacobian (eq. 32)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2],
                          2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                          2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                          2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]])  # (eq. 31)
            J = np.array([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                          [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                          [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                          [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                          [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                          [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx          ]]) # (eq. 32)
            gradient = J.T@f                                        # (eq. 34)
            gradient /= np.linalg.norm(gradient)
            qDot -= self.gain*gradient                              # (eq. 33)
        q += qDot*dt/1000                                           # (eq. 13)
        q /= np.linalg.norm(q)
        return q

degreeToRadian = (2*np.pi)/360

serialPort = serial.Serial(
    port="COM6", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)
serialString = ""  # Used to hold data coming over UART

# test = "13652,4.293,113.797,-1.872,25.376,-378.078,951.722,315.000,-490.000,-385.000,22.898,-0.14587,0.17535,0.40997,23"
# test.split(',')

timeCount = []
point = 0

state = np.eye(3)

xGBias = 0.0011808715524884387*360
yGBias = -0.0012930429755828722*360
zGBias = -0.0008208641842187239*360

while(serialString == b'' or serialString == ""):
    serialString = serialPort.readline().decode('utf-8')
    print(serialString)
serialString = serialPort.readline().decode('utf-8')

# serialString = serialString.decode('utf-8')
print(serialString)
serialList = serialString.split(',')
print(serialList)
lastTime = float(serialList[0])
# print(serialList)
serialList = [float(i) for i in serialList]
time = serialList[0]
voltage = serialList[1]
state_of_charge = serialList[2]
charge_rate = serialList[3]
x_Acc = serialList[4]/1000*9.81 # Convert to m/s^2
y_Acc = serialList[5]/1000*9.81
z_Acc = serialList[6]/1000*9.81
x_Gyro = (serialList[7]/1000 - xGBias) * degreeToRadian  # Convert to rad/s
y_Gyro = (serialList[8]/1000 - yGBias) * degreeToRadian
z_Gyro = (serialList[9]/1000 - zGBias) * degreeToRadian
tempGyro = serialList[10]
x_BField = serialList[11] * 1e-1 # Convert to nT
y_BField = serialList[12] * 1e-1 # Convert to nT
z_BField = serialList[13] * 1e-1 # Convert to nT
tempBField = serialList[14]

a = np.array([x_Acc, y_Acc, z_Acc])
g = np.array([x_Gyro, y_Gyro, z_Gyro])
m = np.array([x_BField, y_BField, z_BField])
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
    
madgwick = MadgwickAltered()

# Initialize global variables
state = np.eye(3)
lastTime = None

def propagate_attitude(state, serialList):
    global lastTime
    time = serialList[0] / 1000  # Convert time to seconds
    xG = (serialList[7] / 1000) - xGBias
    yG = (serialList[8] / 1000) - yGBias
    zG = (serialList[9] / 1000) - zGBias

    if lastTime is None:
        lastTime = time
        return state, np.matmul(state, np.matrix([[0, 0, 1]]).T)

    deltaTime = time - lastTime
    lastTime = time

    # Compute angular velocity vector in radians
    omega = np.array([xG * degreeToRadian, yG * degreeToRadian, zG * degreeToRadian])
    omega_norm = np.linalg.norm(omega)

    if omega_norm > 0:
        # Rodrigues' rotation formula
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        rotation_matrix = (
            np.eye(3) +
            np.sin(omega_norm * deltaTime) / omega_norm * omega_skew +
            (1 - np.cos(omega_norm * deltaTime)) / (omega_norm ** 2) * np.matmul(omega_skew, omega_skew)
        )
        state = np.matmul(rotation_matrix, state)

    # Orthonormalize the state matrix
    state = np.asmatrix(normalize(np.asarray(state), norm='l2', axis=0))

    # Transform acceleration vector
    output_attitude = np.matmul(state, np.matrix([[1, 0, 0]]).T)
    return state, output_attitude

# Initialize buffers for plotting
buffer_size = 500  # Limit the number of points to display
xG_buffer, yG_buffer, zG_buffer = [], [], []
xAttitude_bufferx, yAttitude_bufferx, zAttitude_bufferx = [], [], []
xAttitude_buffery, yAttitude_buffery, zAttitude_buffery = [], [], []
xAttitude_bufferz, yAttitude_bufferz, zAttitude_bufferz = [], [], []
point = 0

def propagate_attitude_madgwick(serialList):
    global madgwick, q, time, lastTime, xGBias, yGBias, zGBias, degreeToRadian

    if lastTime is None:
        lastTime = time
        return state, np.matmul(state, np.matrix([[0, 0, 1]]).T)

    time = serialList[0]
    x_Acc = serialList[4]/9810 # Convert to m/s^2
    y_Acc = serialList[5]/9810
    z_Acc = serialList[6]/9810
    x_Gyro = (serialList[7]/1000 - xGBias) * degreeToRadian  # Convert to rad/s
    y_Gyro = (serialList[8]/1000 - yGBias) * degreeToRadian
    z_Gyro = (serialList[9]/1000 - zGBias) * degreeToRadian
    x_BField = serialList[11] * 1e5 # Convert to nT
    y_BField = serialList[12] * 1e5 # Convert to nT
    z_BField = serialList[13] * 1e5 # Convert to nT

    deltaTime = time - lastTime
    lastTime = time

    g = np.array([x_Gyro, y_Gyro, z_Gyro])
    a = np.array([x_Acc, y_Acc, z_Acc])
    m = np.array([x_BField, y_BField, z_BField])

    # print("madgwick()")
    q = madgwick.updateMARG(q, g, a, m, dt = deltaTime)  # Update the filter with new data
    state = Quaternion(q).to_DCM()  # Convert quaternion to rotation matrix
    # Normalize the rotation matrix
    state = np.asmatrix(normalize(np.asarray(state), norm='l2', axis=0))
    output_attitudex = np.matmul(state, np.matrix([[-1, 0, 0]]).T)
    output_attitudey = np.matmul(state, np.matrix([[0, -1, 0]]).T)
    output_attitudez = np.matmul(state, np.matrix([[0, 0, -1]]).T)
    return state, output_attitudex, output_attitudey, output_attitudez


def animate(i):
    global state, point, xG_buffer, yG_buffer, zG_buffer, xAttitude_bufferx, yAttitude_bufferx, zAttitude_bufferx, xAttitude_buffery, yAttitude_buffery, zAttitude_buffery, xAttitude_bufferz, yAttitude_bufferz, zAttitude_bufferz 
    global lastTime, madgwick, q
    global xGBias, yGBias, zGBias
    try:
        # Flush the serial buffer to get the latest data
        while serialPort.in_waiting > 1:
            serialPort.readline()

        serialString = serialPort.readline().decode('utf-8').strip()
        serialList = [float(x) for x in serialString.split(',')]
        # print(f"Serial List: {serialList}")
        time = serialList[0] / 1000  # Convert time to seconds
        xG = serialList[7] / 1000 - xGBias
        yG = serialList[8] / 1000 - yGBias
        zG = serialList[9] / 1000 - zGBias

        # Propagate attitude
        # print("propagating attitude")
        state, output_attitudex, output_attitudey, output_attitudez = propagate_attitude_madgwick(serialList)
        # Update buffers
        xAttitudex, yAttitudex, zAttitudex = output_attitudex
        xAttitudey, yAttitudey, zAttitudey = output_attitudey
        xAttitudez, yAttitudez, zAttitudez = output_attitudez

        xAttitude_bufferx.append(xAttitudex)
        yAttitude_bufferx.append(yAttitudex)
        zAttitude_bufferx.append(zAttitudex)
        if len(xAttitude_bufferx) > buffer_size:
            xAttitude_bufferx.pop(0)
            yAttitude_bufferx.pop(0)
            zAttitude_bufferx.pop(0)

        xAttitude_buffery.append(xAttitudey)
        yAttitude_buffery.append(yAttitudey)
        zAttitude_buffery.append(zAttitudey)
        if len(xAttitude_bufferx) > buffer_size:
            xAttitude_buffery.pop(0)
            yAttitude_buffery.pop(0)
            zAttitude_buffery.pop(0)

        xAttitude_bufferz.append(xAttitudez)
        yAttitude_bufferz.append(yAttitudez)
        zAttitude_bufferz.append(zAttitudez)
        if len(xAttitude_bufferx) > buffer_size:
            xAttitude_bufferz.pop(0)
            yAttitude_bufferz.pop(0)
            zAttitude_bufferz.pop(0)

        # Update 2D plots
        line1.set_data(range(len(xAttitude_bufferx)), xAttitude_bufferx)
        line2.set_data(range(len(yAttitude_bufferx)), yAttitude_bufferx)
        line3.set_data(range(len(zAttitude_bufferx)), zAttitude_bufferx)

        ax1.set_xlim(0, buffer_size)
        ax2.set_xlim(0, buffer_size)
        ax3.set_xlim(0, buffer_size)
        ax1.set_ylim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax3.set_ylim(-1.1, 1.1)

        # Update 3D plot
        ax4.clear()
        ax4.quiver(0, 0, 0, xAttitudex, yAttitudex, zAttitudex, color='r', label="Attitude")
        ax4.quiver(0, 0, 0, xAttitudey, yAttitudey, zAttitudey, color='g', label="Attitude")
        ax4.quiver(0, 0, 0, xAttitudez, yAttitudez, zAttitudez, color='b', label="Attitude")
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([-1, 1])
        ax4.set_zlim([-1, 1])
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        # ax4.legend()
    except Exception as e:
        print(f"Error in animate: {e}")

# Setup the plot with gridspec
fig = plt.figure(figsize=(10, 10))  # Increase figure size for better visibility
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 2])  # Allocate more space for the 3D plot

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3], projection='3d')  # Add 3D subplot

# Initialize Line2D objects for faster updates
line1, = ax1.plot([], [], lw=1, label="xAttitude")
line2, = ax2.plot([], [], lw=1, label="yAttitude")
line3, = ax3.plot([], [], lw=1, label="zAttitude")
ax4.quiver(0, 0, 0, -1, 0, 0, color='r', label="Attitudex")
ax4.quiver(0, 0, 0, 0, -1, 0, color='r', label="Attitudey")
ax4.quiver(0, 0, 0, 0, 0, -1, color='r', label="Attitudez")

ax1.set_title("xAttitude")
ax2.set_title("yAttitude")
ax3.set_title("zAttitude")
ax1.legend()
ax2.legend()
ax3.legend()

# Adjust layout to prevent overlap and ensure visibility of the 3D plot
# fig.subplots_adjust(hspace=0.5)  # Add more vertical space between subplots

ani = animation.FuncAnimation(fig, animate, interval=1)  # Reduced interval for faster updates
plt.show()