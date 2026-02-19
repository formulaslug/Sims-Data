import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 80        # The port used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.sendall(b'Hello, world')
data = s.recv(1024)

print(f"Received {data.decode()!r}")
s.sendall(b'end protocol')
