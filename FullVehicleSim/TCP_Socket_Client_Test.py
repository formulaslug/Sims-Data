import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 5000        # The port used by the server

    
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.sendall(b'start protocol.R?R')
data = s.recv(1024) ##data gets recieved
print(f"Received {data.decode()!r}")

s.sendall(b'I am still listening!.R?R') 
data = s.recv(1024)
print(f"Received {data.decode()!r}")
s.sendall(b'end protocol')
