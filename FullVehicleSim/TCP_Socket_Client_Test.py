import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 5000        # The port used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def setup():
    s.connect((HOST, PORT))
    s.sendall(b'start protocol')
def close():
    s.shutdown(socket.SHUT_RDWR)
    s.close()
def recieveData():
    data = s.recv(1024) ##data gets recieved
    word = data.decode()
    print(f"Received " + word)
    return word
def sendData(data):
    s.send(data)
def constantly_connect():
    setup()
    while True:
        data = recieveData()
        if(data == "end protocol"):
            sendData('end protocol')
            close()
            break
        strIn = input("Sending Data from client: ")
        sendData(strIn.encode("utf-8"))
constantly_connect()