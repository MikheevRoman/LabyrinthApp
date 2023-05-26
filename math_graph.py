from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt


class GraphView(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Создание сцены и установка ее в QGraphicsView
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Создание вершин графа
        vertex1 = self.create_vertex(50, 50)
        vertex2 = self.create_vertex(150, 50)
        vertex3 = self.create_vertex(100, 150)

        # Создание ребер графа
        self.create_edge(vertex1, vertex2)
        self.create_edge(vertex2, vertex3)
        self.create_edge(vertex3, vertex1)

    vertexes = []

    def get_scene(self):
        return self.scene

    def set_graph(self, coordinates):
        self.vertexes.clear()
        for x_y in coordinates:
            vertex = self.create_vertex(x_y[0], x_y[1])
            self.vertexes.append(vertex)
        # creating edges

    def create_vertex(self, x, y):
        # Создание вершины (эллипса) и добавление его на сцену
        vertex = QGraphicsEllipseItem(x - 10, y - 10, 10, 10)
        vertex.setBrush(QColor(0, 0, 0))  # Цвет вершины
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
