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
def is_connected_to_client():
    try:  
        sock.send(b'some data')
        clientsocket, address = sock.accept()
        data = clientsocket.recv(1024).decode()
        if(data != None):
            print('Server is connected to client!')
            return True
    except:
        return False
def run_socket():
    setup()
    ghost_var = "Velocity data will be here"
    var1 = 0
    clientsocket, address = sock.accept()
    while True:
        print(address)
        data = clientsocket.recv(1024).decode()
        print('Recieved Data at server: ' + data)
        if(data == None or data == 'end protocol'):
            closeSocket()
            break
        phrase = str(input("Input something here: "))
        clientsocket.send(phrase.encode("utf-8"))
    return 0


run_socket()