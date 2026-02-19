import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST_IP = "127.0.0.1" #Made to only use on local machine (for now)
PORT = 80
def setup():
    sock.bind(HOST_IP,PORT)
    sock.listen(5)

def closeSocket():
    return -1

def run_socket():
    while True:
        clientsocket, address = sock.accept()
        
    return 0

