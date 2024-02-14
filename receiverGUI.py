'''
This is the main file for the PyQt GUI application that receives image frames over TCP and displays them.
'''
import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QLabel, QLineEdit
from PyQt5.QtGui import QPixmap, QPainter, QFont, QImage
from PyQt5.QtCore import Qt, pyqtSlot, QByteArray, pyqtSignal, QSize, QTimer 
import numpy as np
from AlgoFCN import FeatureTracker
import cv2
import FNCs

class MainWindow(QMainWindow):
    image_data_signal = pyqtSignal(bytes)
    rcv_frame_counter = 0  # Frame counter to keep track of received frames
    display_frame_counter = 0
    rcvPacketDic = {}
    pixmap = 0

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Frame Receiver')
        self.setGeometry(100, 100, 800, 600)

        self.server_socket = None
        self.client_socket = None
        self.image_data_signal.connect(self.recv_image)
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        self.port_input = QLineEdit(self)
        self.port_input.setText('65432')
        self.start_button = QPushButton('Start Server', self)
        self.start_button.clicked.connect(self.start_server)

        # empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # _, testbuffer = cv2.imencode('.png', empty_image)
        # self.imgData = testbuffer.tobytes()
        buffer = np.full((200, 200, 3), fill_value=255, dtype=np.uint8)
        image = QImage(buffer.data, buffer.shape[1], buffer.shape[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.tracker = FeatureTracker()

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.port_input)
        layout.addWidget(self.start_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
    @pyqtSlot()
    def start_server(self):
        port = int(self.port_input.text())
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', port))  # Bind to all interfaces on the specified port
        self.server_socket.listen(1)
        print(f"Server listening on port {port}")
        threading.Thread(target=self.server_thread, daemon=True).start()
        self.start_timer()

    @pyqtSlot(bytes)
    def recv_image(self, image_data):

        __cpTracks, __kpMvmt = self.tracker.process_frame(image_data, self.rcv_frame_counter)
        self.rcvPacketDic[self.rcv_frame_counter] = (__cpTracks, __kpMvmt, image_data)
        self.rcv_frame_counter +=1

    def start_timer(self):
        self.timer.start(1000)  # Timer set to 1 second for demonstration purposes
    
    def update_image(self):
        __cpTracks, __kpMvmt, __imgData = {}, {}, 0
        if self.display_frame_counter in self.rcvPacketDic:
            __cpTracks, __kpMvmt, __imgData = self.rcvPacketDic[self.display_frame_counter]
        else:
            return

        imgcp = self.update_cvFrame(__cpTracks, __imgData, self.display_frame_counter)
                # Conversion: Encode cvFrame to a format (e.g., PNG) and then convert to QByteArray
        _, buffer = cv2.imencode('.png', imgcp)  # Encode the image as PNG; adjust format as needed
        image_data_qba = QByteArray(buffer.tobytes())  # Convert buffer to QByteArray
        self.pixmap = QPixmap()
        self.pixmap.loadFromData(image_data_qba)

        # Draw frame number on the image
        painter = QPainter(self.pixmap)
        painter.setFont(QFont('Arial', 20))
        painter.drawText(self.pixmap.rect(), Qt.AlignBottom | Qt.AlignRight, str(self.display_frame_counter))
        painter.end()
        
        self.image_label.setPixmap(self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Increment frame counter
        self.display_frame_counter += 1
        return
    
    def update_cvFrame(self, _cpTracks, _frameData, _crnt_frm_idx):
        _cvFrame = cv2.imdecode(np.frombuffer(_frameData, np.uint8), cv2.IMREAD_COLOR)

        # First, filter out tracks with less than 3 keypoints to avoid modifying the dictionary during iteration
        _cpTracks = {key: value for key, value in _cpTracks.items() if len(value) >= 3}

        # Now, process the tracks for drawing
        for _track_id, _keypoints in _cpTracks.items():
            # Pre-compute the existence of the current frame in keypoints to avoid repeated checks
            current_frame_exists = any(f == _crnt_frm_idx for f, _ in _keypoints)
            
            # If the current frame doesn't exist in the keypoints, skip further processing for this track
            if not current_frame_exists:
                continue

            # Draw each track on the frame
            for f, point in _keypoints:
                if f == _crnt_frm_idx:  # Check if the keypoint belongs to the current frame
                    cv2.circle(_cvFrame, point, 5, (0, 255, 0), -1)  # Draw the keypoint

            # Draw lines between keypoints of the same track
            for kp_idx in range(1, len(_keypoints)):
                f, point = _keypoints[kp_idx]
                if f <= _crnt_frm_idx:
                    start_point = _keypoints[kp_idx-1][1]
                    end_point = point
                    cv2.line(_cvFrame, start_point, end_point, (255, 0, 0), 2)
                
        return _cvFrame
    
    def server_thread(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=self.client_handler, args=(client_socket,), daemon=True).start()
    
    def client_handler(self, client_socket):
        try:
            while True:
                # Assuming the client sends the size of the image first (4 bytes)
                size_info = client_socket.recv(4)
                if not size_info:
                    break
                size = int.from_bytes(size_info, byteorder='big')
                image_data = b''
                while len(image_data) < size:
                    packet = client_socket.recv(size - len(image_data))
                    if not packet:
                        return  # Connection closed
                    image_data += packet
                self.image_data_signal.emit(image_data)
        finally:
            client_socket.close()
    
    def resizeEvent(self, event):
        # Allow the image_label to shrink by setting its minimum size to a smaller value
        self.image_label.setMinimumSize(1, 1)  # Set minimum size to almost zero
        
        # Calculate new size considering padding/margins if necessary
        newWidth = max(self.width() - 160, 1)  # Ensure width is at least 1
        newHeight = max(self.height() - 120, 1)  # Ensure height is at least 1
        self.image_label.resize(newWidth, newHeight)
        
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())