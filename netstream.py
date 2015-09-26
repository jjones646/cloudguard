import socket
import time
import picamera
import datetime as dt

BIND_ADDR = '0.0.0.0'
BIND_PORT = 8000

print "--  server starting on ", ("%s:%d") % (BIND_ADDR, BIND_PORT)
print "--  use Ctrl-D to quit\n"

with picamera.PiCamera() as camera:
    while True:
        try:
            camera.resolution = (640, 480)
            camera.framerate = 16
            camera.annotate_background = picamera.Color("black")

            server_socket = socket.socket()
            server_socket.bind((BIND_ADDR, BIND_PORT))
            server_socket.listen(0)

            try:
                # Accept a single connection and make a file-like object out of it
                connection = server_socket.accept()[0]
                stream = connection.makefile('wb')
                           
                camera.annotate_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                camera.start_recording(stream, format='h264')
                print "--  connection from", "%s:%d"  % connection.getpeername()
                camera.wait_recording(60)
                camera.stop_recording()

            finally:
                #camera.close() 
                stream.close()
                connection.close()

        finally:
            print "--  closed"
            #server_socket.close()
