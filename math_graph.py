from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt

import algorythms


class GraphView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Создание вершин графа
        # vertex1 = self.create_vertex(50, 50)
        # vertex2 = self.create_vertex(150, 50)
        # vertex3 = self.create_vertex(100, 150)
        #
        # # Создание ребер графа
        # self.create_edge(vertex1, vertex2)
        # self.create_edge(vertex2, vertex3)
        # self.create_edge(vertex3, vertex1)

    maze = [[bool]]
    vertexes = [[]]
    edges = []
    adjacency_matrix = [[]]

    def get_scene(self):
        return self.scene

    # def set_maze(self, maze):
    #     self.maze = maze

    def set_adjacency_matrix(self, matrix):
        self.adjacency_matrix = matrix

    def set_graph(self, maze):
        self.maze = maze
        self.adjacency_matrix = algorythms.transform_to_adjacency_table(self.maze)
        edges_list = algorythms.convert_adjacency_matrix_for_vertexes(self.adjacency_matrix)

        y_size = len(maze)
        x_size = len(maze[0])
        cell_size = max(20 * 30 / max(x_size, y_size), 15)
        # pos = []
        for row in range(y_size):
            for col in range(x_size):
                if not maze[row][col]:
                    # pos.append((col * cell_size, row * cell_size))
                    vertex = self.create_vertex(col * cell_size, row * cell_size)
                    index = row * y_size + col

                    self.vertexes.append([vertex, index])

        for edge in edges_list:
            self.create_edge(self.vertexes[edge[0]][1], self.vertexes[edge[1]][1])

        # self.vertexes.clear()
        # for x_y in coordinates:
        #     vertex = self.create_vertex(x_y[0], x_y[1])
        #     vertex.id = len(self.vertexes)
        #     self.vertexes.append(vertex)
        # # creating edges
        # pass

    def create_vertex(self, x, y):
        # Создание вершины (эллипса) и добавление его на сцену
        vertex = QGraphicsEllipseItem(x - 10, y - 10, 10, 10)
        vertex.setBrush(QColor(0, 0, 0))
        self.scene.addItem(vertex)
        return vertex

    def create_edge(self, start, end):
        # Создание ребра (линии) между двумя вершинами и добавление его на сцену
        edge = QGraphicsLineItem()
        edge.setLine(start.rect().center().x(), start.rect().center().y(),
                     end.rect().center().x(), end.rect().center().y())
        pen = QPen(Qt.black)
        pen.setWidth(2)
        edge.setPen(pen)
        self.scene.addItem(edge)


if __name__ == '__main__':
    app = QApplication([])
    view = GraphView()
    view.show()
    app.exec_()
