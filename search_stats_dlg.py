import time
import numpy
import algorythms as al
from PyQt5.QtWidgets import QDialog
from search_stats_dlg_ui import Ui_Dialog
from test_pyqtgraph import GraphWidget


class SearchStatsDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.collect_stats_btn.clicked.connect(lambda: self.collect_statistics())

    maze_solve_algorythms = [al.ai_lab_solve, al.a_star, al.dijkstra, al.bfs]
    maze_gen_algorythms = [al.growing_tree, al.aldous_broder, al.wilson, al.backtracking, al.binary_tree, al.kruskal,
                           al.modified_prim, al.sidewinder, al.division, al.serpentine, al.small_rooms, al.spiral]
    stats = []  # list of data
    graph_size = []  # list of lab size

    def collect_statistics(self):
        # if ()
        start = int(self.initial_dimention_le.text())
        end = int(self.finite_dimention_le.text())
        step = int(self.steps_number_le.text())
        step = int((end - start) / step)
        if step % 2 != 0:
            step += 1
        print(step)
        count = int(self.repeat_number_le.text())
        solve_al_index = self.solveMethod_comboBox.currentIndex()
        gen_al_index = self.generationMethod_comboBox.currentIndex()

        if start == end:
            self.graph_size = [start]
        else:
            self.graph_size = numpy.arange(start, end + 1, step)
        print(self.graph_size)

        for size in self.graph_size:
            sum_time = 0.0
            for i in range(count):
                maze = self.maze_gen_algorythms[gen_al_index](size, size)
                start_time = time.time()
                self.maze_solve_algorythms[solve_al_index](maze, (1, 1), (len(maze) - 2, len(maze[0]) - 2))
                end_time = time.time()
                dif_time = end_time - start_time
                sum_time += dif_time
            sum_time /= count
            self.stats.append(sum_time)
            print(sum_time)

        print(self.stats)
        self.show_graph()

    def show_graph(self):
        self.graph = GraphWidget()
        self.graph.setWindowTitle("Графики характеристик решения")
        self.graph.add_plot(self.graph_size, self.stats, "Зависимость времени прохождения от размера",
                            "Размер лабиринта", "Время прохождения")
        self.graph.add_plot(self.graph_size, self.stats, "Зависимость времени прохождения от размера",
                            "Размер лабиринта", "Количество шагов")
        self.graph.show()
