import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST_IP = "127.0.0.1" #Made to only use on local machine (for now)
PORT = 80
def setup():
    sock.bind((HOST_IP,PORT))
    sock.listen(5)

def closeSocket():
    sock.shutdown()
    sock.close()

def run_socket():
    setup()
    ghost_var = "Velocity data will be here"
    while True:
        clientsocket, address = sock.accept()
        print(address)
        data = clientsocket.recv(1024).decode()
        if(data == None or data == 'end protocol'):
            closeSocket()
            break
        clientsocket.send(ghost_var.encode("utf-8"))
    return 0


run_socket()