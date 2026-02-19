import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST_IP = "127.0.0.1" #Made to only use on local machine (for now)
PORT = 80
def setup():
    sock.bind(HOST_IP,PORT)
    sock.listen(5)

def closeSocket():
    sock.shutdown()
    sock.close()

def run_socket():
    ghost_var = "Velocity data will be here"
    while True:
        clientsocket, address = sock.accept()
        data = clientsocket.recv(1024)
        if(data == None):
            closeSocket()
        clientsocket.send(ghost_var)
    return 0

