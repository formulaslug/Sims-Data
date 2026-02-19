from threading import Thread
import subprocess

t1 = Thread(target=subprocess.run, args=(["python", "FullVehicleSim\TCP_Socket_Server.py"],))
t2 = Thread(target=subprocess.run, args=(["python", "FullVehicleSim\TCP_Socket_Client_Test.py"],))

t1.start()
t2.start()

t1.join()
t2.join()
