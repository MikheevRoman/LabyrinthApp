import time
from PyQt5.QtWidgets import QDialog
from test_pyqtgraph import GraphWidget
from generation_stats_dlg_ui import Ui_Dialog
import algorythms as al
import numpy


class GenerationStatsDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.collect_stats_btn.clicked.connect(lambda: self.collect_statistics())

    maze_gen_algorythms = [al.growing_tree, al.aldous_broder, al.wilson, al.backtracking, al.binary_tree, al.kruskal,
                           al.modified_prim, al.sidewinder, al.division, al.serpentine, al.small_rooms, al.spiral]
    stats = []  # list of data
    graph_size = []  # list of lab size
    dead_ends_stat = []  # list of dead ends

    def collect_statistics(self):
        start = int(self.initial_dimention_le.text())
        end = int(self.finite_dimention_le.text())
        # вычисляем шаг
        step = int(self.steps_number_le.text())
        step = int((end - start) / step)
        if step % 2 != 0:
            step += 1
        count = int(self.repeat_number_le.text())
        index = self.generationMethod_comboBox.currentIndex()

        if start == end:
            self.graph_size = [start]
        else:
            self.graph_size = numpy.arange(start, end + 1, step)
        print(self.graph_size)

        for size in self.graph_size:
            sum_time = 0.0
            sum_dead_ends = 0.0
            for i in range(count):
                start_time = time.time()
                temp = self.maze_gen_algorythms[index](size, size)
                end_time = time.time()
                dif_time = end_time - start_time
                sum_time += dif_time
                sum_dead_ends += al.num_of_dead_ends(temp) / (size * size) * 100  # определение количества тупиков
            sum_time /= count
            sum_dead_ends /= count
            self.stats.append(sum_time)
            self.dead_ends_stat.append(sum_dead_ends)
            print(sum_time, sum_dead_ends)

        print(self.stats)
        self.show_graph()

    def show_graph(self):
        self.graph = GraphWidget()
        self.graph.showMaximized()
        self.graph.setWindowTitle("Графики генерации")
        self.graph.add_plot(self.graph_size, self.stats, "Зависимость времени генерации от размера", "Размер лабиринта (NxN ячеек)", "Время генерации (мс)")
        self.graph.add_plot(self.graph_size, self.dead_ends_stat, "Зависимость количества тупиков от размера", "Размер лабиринта (NxN ячеек)", "Процент тупиков")
        self.graph.show()
        self.close()
