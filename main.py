import time
from math_graph import GraphView
from PyQt5 import QtWidgets, QtGui, QtCore, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem, QDialog, QVBoxLayout, QLabel, \
    QPushButton, QMainWindow, QMessageBox, QGraphicsItem
import algorythms
from mainwindow import Ui_MainWindow
from generation_stats_dlg import GenerationStatsDialog
from search_stats_dlg import SearchStatsDialog


class ClickableRectItem(QGraphicsRectItem):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        brush = QColor(250, 250, 250)
        self.setBrush(brush)  # Начальный цвет объекта

    def mousePressEvent(self, event):
        if event.button() == 1:  # ЛКМ
            if self.brush() == QColor(250, 250, 250):
                brush = QColor(0, 255, 0)
                self.setBrush(brush)
                self.update()
                MainWindow.start_point = self.index
                print(self.index)
        if event.button() == 2:  # ПКМ
            if self.brush() == QColor(250, 250, 250):
                brush = QColor(255, 0, 0)
                self.setBrush(brush)
                self.update()
                MainWindow.end_point = self.index
                print(self.index)
        else:
            super().mousePressEvent(event)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.showMaximized()

        self.generateBtn.clicked.connect(lambda: self.generate_labyrinth())
        self.solve_btn.clicked.connect(lambda: self.show_path())
        self.generation_stat_collection_btn.clicked.connect(lambda: self.generation_stat_collection())
        self.solution_stat_collect_btn.clicked.connect(lambda: self.solution_stat_collection())
        self.fullscreen_set_action.triggered.connect(lambda: self.showFullScreen())
        self.fullscreen_exit_action.triggered.connect(lambda: self.showMaximized())
        self.clear_maze_btn.clicked.connect(lambda: self.clear_maze())
        self.convert_to_graph_btn.clicked.connect(lambda: self.convert_to_graph())
        self.about_action.triggered.connect(lambda: QMessageBox.about(self, "О программе", "Программа разработа в рамках научно-исследовательской работы студентами факультета ИКСС СПбГУТ"))

    maze_gen_algorythms = [algorythms.growing_tree, algorythms.aldous_broder, algorythms.wilson, algorythms.backtracking, algorythms.binary_tree, algorythms.kruskal, algorythms.eller,
                           algorythms.modified_prim, algorythms.sidewinder, algorythms.division, algorythms.serpentine, algorythms.small_rooms, algorythms.spiral]
    maze_solve_algorythms = [algorythms.ai_lab_solve, algorythms.a_star, algorythms.dijkstra, algorythms.bfs]
    global_grid = []
    generation_time = 0.0
    search_time = 0.0
    start_point = 0
    end_point = 0
    x_size = 0
    y_size = 0

    def solution_stat_collection(self):
        dlg = SearchStatsDialog()
        dlg.exec()

    def generation_stat_collection(self):
        dlg = GenerationStatsDialog()
        dlg.exec()

    def generate_labyrinth(self):
        global grid, start_time
        if (self.y_lineEdit.text() != '') and (self.x_lineEdit.text() != '') \
                and (int(self.y_lineEdit.text()) >= 3) and (int(self.y_lineEdit.text()) >= 3) and (int(self.y_lineEdit.text()) % 2 != 0) and (int(self.x_lineEdit.text()) % 2 != 0):
            self.y_size = int(self.y_lineEdit.text())
            self.x_size = int(self.x_lineEdit.text())

            if self.generationMethod_comboBox.currentIndex() == 0:
                QMessageBox.warning(self, "Ошибка", "Выберите способ генерации лабиринта")
                return
            start_time = time.time()
            grid = self.maze_gen_algorythms[self.generationMethod_comboBox.currentIndex() - 1](self.y_size, self.x_size)
            self.generation_time = time.time() - start_time

            self.global_grid = grid
            scene = QGraphicsScene()
            self.graphicsView.setScene(scene)
            # Создаем прямоугольник для каждой ячейки лабиринта
            cell_size = max(20 * 30/max(self.x_size, self.y_size), 15)
            index = 0
            for row in range(self.y_size):
                for col in range(self.x_size):
                    rect = ClickableRectItem(index)
                    rect.setRect(col * cell_size, row * cell_size, cell_size, cell_size)
                    if grid[row][col]:
                        brush = QColor(0, 0, 0)
                        rect.setBrush(brush)
                    scene.addItem(rect)
                    index += 1
        else:
            QMessageBox.warning(self, "Ошибка", "Ширина и длина лабиринта должны быть >=4 и принимать нечетные значения")

    def convert_to_graph(self):
        self.y_size = int(self.y_lineEdit.text())
        self.x_size = int(self.x_lineEdit.text())
        cell_size = max(20 * 30 / max(self.x_size, self.y_size), 15)
        pos = []
        for row in range(self.y_size):
            for col in range(self.x_size):
                if not self.global_grid[row][col]:
                    pos.append((col * cell_size, row * cell_size))

        matrix = algorythms.transform_to_adjacency_table(self.global_grid)

        graph = GraphView()
        graph.set_graph(self.global_grid)
        #
        #   ...
        #
        self.graphicsView.setScene(graph.get_scene())

        self.convert_to_graph_btn.setText("Преобразовать в лабиринт")
        self.convert_to_graph_btn.clicked.disconnect()
        self.convert_to_graph_btn.clicked.connect(lambda: self.show_labyrinth())

    def show_labyrinth(self):
        cell_size = max(20 * 30 / max(self.x_size, self.y_size), 15)
        index = 0
        scene = QGraphicsScene()
        for row in range(self.y_size):
            for col in range(self.x_size):
                rect = ClickableRectItem(index)
                rect.setRect(col * cell_size, row * cell_size, cell_size, cell_size)
                if self.global_grid[row][col]:
                    brush = QColor(0, 0, 0)
                    rect.setBrush(brush)
                scene.addItem(rect)
                index += 1
        self.graphicsView.setScene(scene)

        self.convert_to_graph_btn.setText("Преобразовать в граф")
        self.convert_to_graph_btn.clicked.disconnect()
        self.convert_to_graph_btn.clicked.connect(lambda: self.convert_to_graph())

    def show_path(self):
        # len(self.global_grid[0] - x
        if self.start_point > 0 and self.end_point > 0:
            start = (self.start_point // len(self.global_grid[0]), self.start_point % len(self.global_grid[0]))
            end = (self.end_point // len(self.global_grid[0]), self.end_point % len(self.global_grid[0]))
        else:
            start = (1, 1)
            end = (len(self.global_grid) - 2, len(self.global_grid[0]) - 2)

        method = self.solutionMethod.currentIndex()
        if method == 0:
            QMessageBox.warning(self, "Ошибка", "Выберите способ решения")
            return
        path = self.maze_solve_algorythms[method - 1](self.global_grid, start, end)

        newbie_grid = []
        scene_grid = self.graphicsView.items()
        cell_size = max(20 * 30 / max(len(self.global_grid[0]), len(self.global_grid)), 15)
        for index in path:
            rect = QGraphicsRectItem(index[1] * cell_size, index[0] * cell_size, cell_size, cell_size)
            brush = QColor(0, 0, 255)
            rect.setBrush(brush)
            scene_grid.append(rect)
            newbie_grid.append(rect)

        scene = self.graphicsView.scene()
        for item in newbie_grid:
            scene.addItem(item)

    def clear_maze(self):
        self.x_size = len(self.global_grid[0])
        self.y_size = len(self.global_grid)
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        cell_size = max(20 * 30 / max(self.x_size, self.y_size), 15)
        index = 0
        for row in range(self.y_size):
            for col in range(self.x_size):
                rect = ClickableRectItem(index)
                rect.setRect(col * cell_size, row * cell_size, cell_size, cell_size)
                if grid[row][col]:
                    brush = QColor(0, 0, 0)
                    rect.setBrush(brush)
                scene.addItem(rect)
                index += 1


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
