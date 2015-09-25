import socket
import time
import picamera

BIND_ADDR = '0.0.0.0'
BIND_PORT = 8000

print "--  server starting on ", ("%s:%d") % (BIND_ADDR, BIND_PORT)
print "--  use Ctrl-D to quit"

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 16

    server_socket = socket.socket()
    server_socket.bind((BIND_ADDR, BIND_PORT))
    server_socket.listen(0)

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('wb')

    try:
        camera.start_recording(connection, format='h264')
        camera.wait_recording(60)
        camera.stop_recording()

    finally:
        connection.close()
        server_socket.close()
