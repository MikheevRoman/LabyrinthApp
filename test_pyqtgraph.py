import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel


class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графики")
        self.showMaximized()
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        self.setLayout(layout)

    def add_plot(self, x, y, title="", axis_x_title="", axis_y_title=""):
        new_layout = self.layout()
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title, color=(0, 0, 0))
        styles = {'color': 'g', 'font-size': '16px'}
        plot_widget.setLabel('left', axis_y_title, **styles)
        plot_widget.setLabel('bottom', axis_x_title, **styles)
        plot_widget.setBackground((227, 227, 227))
        pen = pg.mkPen(color=(31, 17, 140), width=2)
        plot_widget.plot(x, y, pen=pen)
        new_layout.addWidget(plot_widget)
        self.setLayout(new_layout)
