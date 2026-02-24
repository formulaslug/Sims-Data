import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST_IP = "127.0.0.1" #Made to only use on local machine (for now)
PORT = 5000
def setup():
    sock.bind((HOST_IP,PORT))
    sock.listen(5)

def closeSocket():
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()

def run_socket():
    setup()
    ghost_var = "Velocity data will be here"
    var1 = 0
    
    while True:
        print(address)
        clientsocket, address = sock.accept()
        data = clientsocket.recv(1024).decode()
         ##Code gets stuck on here since it blocks?
        print('Recieved Data: ' + data)
        if(data == None or data == 'end protocol'):
            closeSocket()
            break
        if(var1 == 0):
            clientsocket.send(ghost_var.encode("utf-8"))
            print('sent fake velocity data')
            var1 +=1
    return 0


run_socket()