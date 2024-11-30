'''
This is the main file for the PyQt GUI application that receives image frames over TCP and displays them.
'''
import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFormLayout, \
        QGridLayout, QSpacerItem ,QMainWindow, QPushButton, QLabel, QLineEdit, QDoubleSpinBox
from PyQt5.QtGui import QPixmap, QPainter, QFont, QImage, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, pyqtSlot, QByteArray, pyqtSignal, QSize, QTimer, QRectF
import numpy as np
from AlgoFCN import FeatureTracker
from QCustomPlot_PyQt5 import *  # Make sure this import works
import cv2
import FNCs
import math

class MainWindow(QMainWindow):
    image_data_signal = pyqtSignal(bytes)
    rcv_frame_counter = 0  # Frame counter to keep track of received frames
    display_frame_counter = 0
    rcvPacketDic = {}
    pixmap = 0
    desired_coords = [0, 0, 640, 480] # [x_min, y_min, x_max, y_max]

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Frame Receiver')
        self.setGeometry(100, 100, 1200, 600)  # Adjusted window size to accommodate plot

        self.server_socket = None
        self.client_socket = None
        self.image_data_signal.connect(self.recv_image)
        self.image_label = QLabel(self)
        self.image_label.resize(self.desired_coords[2], self.desired_coords[3])
        self.port_input = QLineEdit(self)
        self.port_input.setText('65432')
        self.start_button = QPushButton('Start Server', self)
        self.start_button.clicked.connect(self.start_server)

        self.x_min_label = QLabel("X Min:", self)
        self.x_min_spinbox = QDoubleSpinBox(self)
        self.y_min_label = QLabel("Y Min:", self)
        self.y_min_spinbox = QDoubleSpinBox(self)
        self.x_max_label = QLabel("X Max:", self)
        self.x_max_spinbox = QDoubleSpinBox(self)
        self.y_max_label = QLabel("Y Max:", self)
        self.y_max_spinbox = QDoubleSpinBox(self)

        self.init_desired_coords_GUI()

        self.horizontalSpacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # self.horizontalSpacer = QSpacerItem(self)

        self.tracker = FeatureTracker()

        # Set up the custom plot (from the QCustomPlot example)
        self.xData = []
        self.yData = []
        self.MAX_LEN_PLOT_X_AXES = 100
        self.m_vecPlot_X = []
        self.m_vecPlot_Y = []
        self.customPlot = QCustomPlot()
        graph0 = self.customPlot.addGraph()
        graph0.setPen(QPen(Qt.blue))
        graph0.setBrush(QBrush(QColor(0, 0, 255, 20)))
        self.customPlot.xAxis.setRange(0, self.MAX_LEN_PLOT_X_AXES)
        self.customPlot.yAxis.setRange(0, 7)
        self.customPlot.axisRect().insetLayout().setInsetAlignment(0, Qt.AlignmentFlag.AlignLeft)
        self.customPlot.graph(0).setLineStyle(QCPGraph.lsNone)
        self.customPlot.graph(0).setScatterStyle(QCPScatterStyle(QCPScatterStyle.ScatterShape.ssDisc, 10))
        self.customPlot.rescaleAxes()

        buffer = np.full((200, 200, 3), fill_value=255, dtype=np.uint8)
        image = QImage(buffer.data, buffer.shape[1], buffer.shape[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)

        # Layout adjustments
        mainLayout = QVBoxLayout()
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.image_label, 1)
        topLayout.addWidget(self.customPlot, 1)  # Add the custom plot to the layout
        mainLayout.addLayout(topLayout)
        
        controlLayout = QVBoxLayout()
        controlLayout.addWidget(self.port_input)
        controlLayout.addWidget(self.start_button)
        # # Convert to QFormLayout for the new DoubleSpinBoxes and Labels
        # formLayout = QFormLayout()
        # formLayout.addRow(self.x_min_label, self.x_min_spinbox)
        # formLayout.addRow(self.y_min_label, self.y_min_spinbox)
        # formLayout.addRow(self.x_max_label, self.x_max_spinbox)
        # formLayout.addRow(self.y_max_label, self.y_max_spinbox)

        gridLayout = QGridLayout()
        # First row
        gridLayout.addWidget(self.x_min_label, 0, 0)  # Row 0, Column 0
        gridLayout.addWidget(self.x_min_spinbox, 0, 1)  # Row 0, Column 1
        gridLayout.addItem(self.horizontalSpacer, 0, 2)  # Horizontal spacer 0, 2
        gridLayout.addWidget(self.y_min_label, 0, 3)  # Row 0, Column 2
        gridLayout.addWidget(self.y_min_spinbox, 0, 4)  # Row 0, Column 3
        # Second 
        gridLayout.addWidget(self.x_max_label, 1, 0)  # Row 1, Column 0
        gridLayout.addWidget(self.x_max_spinbox, 1, 1)  # Row 1, Column 1
        gridLayout.addItem(self.horizontalSpacer, 1, 2)  # Horizontal spacer 0, 2
        gridLayout.addWidget(self.y_max_label, 1, 3)  # Row 1, Column 2
        gridLayout.addWidget(self.y_max_spinbox, 1, 4)  # Row 1, Column 3

        ButtomLayout = QHBoxLayout()
        ButtomLayout.addLayout(controlLayout, 1)
        ButtomLayout.addLayout(gridLayout, 1)
        # # You can then add this formLayout to your existing controlLayout or mainLayout as needed
        # controlLayout.addLayout(formLayout)

        mainLayout.addLayout(ButtomLayout)
        container = QWidget()
        container.setLayout(mainLayout)
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

        self.desired_coords = [int(self.x_min_spinbox.value() * self.pixmap.width()),
                               int(self.y_min_spinbox.value() * self.pixmap.height()),
                               int(self.x_max_spinbox.value() * self.pixmap.width()),
                               int(self.y_max_spinbox.value() * self.pixmap.height())]

        tracks, kp_loxy_list, image_cv2, _frmTrackIds, _ = self.tracker.process_frame(image_data, self.rcv_frame_counter, self.desired_coords, True)

        # cvFrame = self.tracker.update_cvFrame(self.rcv_frame_counter, tracks, image_cv2, _frmTracks, self.desired_coords, _isOnline = False)

        self.rcvPacketDic[self.rcv_frame_counter] = (tracks, kp_loxy_list, image_cv2, _frmTrackIds)

        growthRate = np.mean([item[3]*5 for item in kp_loxy_list])
        if growthRate > 0:
            growthRate = np.log(growthRate) #for better visualization
        else:
            growthRate = 0
        self.update_gr_plot_data(self.rcv_frame_counter, growthRate) # kaveh: it is important to update the plot data here
        self.rcv_frame_counter +=1

    def start_timer(self):
        self.timer.start(1000)  # Timer set to 1 second for demonstration purposes
    
    def update_image(self):
        tracks, image_cv2, _frmTrackIds, kp_loxy_list = None, None, None, None
        if self.display_frame_counter in self.rcvPacketDic:
            tracks, kp_loxy_list, image_cv2, _frmTrackIds = self.rcvPacketDic[self.display_frame_counter]
        else:
            return
        
        # Update the plot
        self.customPlot.graph(0).setData(self.m_vecPlot_X, self.m_vecPlot_Y)
        if self.rcv_frame_counter > self.MAX_LEN_PLOT_X_AXES: 
            self.customPlot.xAxis.setRange(self.rcv_frame_counter - self.MAX_LEN_PLOT_X_AXES, self.rcv_frame_counter)
        self.customPlot.replot()

        # Process the frame and get the processed image
        imgcp = self.tracker.update_cvFrame(self.rcv_frame_counter, tracks, image_cv2, _frmTrackIds, self.desired_coords, _isOnline = True) 
        # imgcp = self.update_cvFrame(__cpTracks, __imgData, self.display_frame_counter)
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
        
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
        painter.drawRect(QRectF(self.desired_coords[0],
                                 self.desired_coords[1],
                                 self.desired_coords[2] - self.desired_coords[0],
                                 self.desired_coords[3] - self.desired_coords[1]))
        painter.end()

        self.image_label.setPixmap(self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Increment frame counter
        self.display_frame_counter += 1
        return
    
    def update_gr_plot_data(self, frameNo, growthRate):
        # Add new data point
        self.xData.append(frameNo)  # Assuming display_frame_counter is your x-value

        self.yData.append(growthRate) 
        # Update the plot data
        if len(self.xData) >= self.MAX_LEN_PLOT_X_AXES:
            # Remove the oldest data point
            self.xData.pop(0)
            self.yData.pop(0)

        self.m_vecPlot_X = self.xData
        self.m_vecPlot_Y = self.yData

    def update_cvFrame(self, _cpTracks, _frameData, _crnt_frm_idx):
        _cvFrame = cv2.imdecode(np.frombuffer(_frameData, np.uint8), cv2.IMREAD_COLOR)

        # First, filter out tracks with less than 3 keypoints to avoid modifying the dictionary during iteration
        _cpTracks = {key: value for key, value in _cpTracks.items() if len(value) >= 3}

        # translation
        xt, yt = self.desired_coords[0], self.desired_coords[1]
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
                    t_point = (point[0] + xt, point[1] + yt)  # Translate the point
                    cv2.circle(_cvFrame, t_point, 5, (0, 255, 0), -1)  # Draw the keypoint

            # Draw lines between keypoints of the same track
            for kp_idx in range(1, len(_keypoints)):
                f, point = _keypoints[kp_idx]
                # if f <= _crnt_frm_idx:
                #     start_point = _keypoints[kp_idx-1][1]
                #     end_point = point
                #     cv2.line(_cvFrame, start_point, end_point, (255, 0, 0), 2)
                if f <= _crnt_frm_idx:
                    start_point = (_keypoints[kp_idx-1][1][0] + xt, _keypoints[kp_idx-1][1][1] + yt)  # Translate the start point
                    end_point = (point[0] + xt, point[1] + yt)  # Translate the end point
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
                # 4 bytes for size, 4 bytes for width, 4 bytes for height
                header_info = client_socket.recv(12)
                if not header_info:
                    break

                # Extract size, width, and height from the received bytes
                size = int.from_bytes(header_info[:4], byteorder='big')
                frame_width = int.from_bytes(header_info[4:8], byteorder='big')
                frame_height = int.from_bytes(header_info[8:12], byteorder='big')

                print(f"Size: {size}, Width: {frame_width}, Height: {frame_height}")

                image_data = bytearray()
                bytes_received = 0
                while bytes_received < size:
                    packet = client_socket.recv(min(4096, size - bytes_received))
                    if not packet:
                        return  # Connection closed
                    image_data.extend(packet)
                    bytes_received += len(packet)

                self.image_data_signal.emit(bytes(image_data))

        except Exception as e:
            print(e)
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

    def init_desired_coords_GUI(self):

        self.x_min_spinbox.setDecimals(1)       
        self.y_min_spinbox.setDecimals(1)       
        self.x_max_spinbox.setDecimals(1)       
        self.y_max_spinbox.setDecimals(1)
        
        self.x_min_spinbox.setRange(0, 0.4)        
        self.y_min_spinbox.setRange(0, 0.4)        
        self.x_max_spinbox.setRange(0.5, 1)        
        self.y_max_spinbox.setRange(0.5, 1) 

        self.x_min_spinbox.setSingleStep(0.05)   
        self.y_min_spinbox.setSingleStep(0.05)      
        self.x_max_spinbox.setSingleStep(0.05)        
        self.y_max_spinbox.setSingleStep(0.05)   

        self.x_min_spinbox.setValue(0)        
        self.y_min_spinbox.setValue(0)
        self.x_max_spinbox.setValue(1)        
        self.y_max_spinbox.setValue(0.5) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())