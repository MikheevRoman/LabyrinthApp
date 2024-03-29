# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(900, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 325, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 7, 1, 1, 2)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 3, 13, 1)
        self.generateBtn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.generateBtn.sizePolicy().hasHeightForWidth())
        self.generateBtn.setSizePolicy(sizePolicy)
        self.generateBtn.setDefault(True)
        self.generateBtn.setObjectName("generateBtn")
        self.gridLayout.addWidget(self.generateBtn, 4, 1, 1, 2)
        self.solution_stat_collect_btn = QtWidgets.QPushButton(self.centralwidget)
        self.solution_stat_collect_btn.setObjectName("solution_stat_collect_btn")
        self.gridLayout.addWidget(self.solution_stat_collect_btn, 11, 1, 1, 2)
        self.generation_stat_collection_btn = QtWidgets.QPushButton(self.centralwidget)
        self.generation_stat_collection_btn.setObjectName("generation_stat_collection_btn")
        self.gridLayout.addWidget(self.generation_stat_collection_btn, 6, 1, 1, 2)
        self.solve_btn = QtWidgets.QPushButton(self.centralwidget)
        self.solve_btn.setObjectName("solve_btn")
        self.gridLayout.addWidget(self.solve_btn, 10, 1, 1, 2)
        self.generationMethod_comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.generationMethod_comboBox.sizePolicy().hasHeightForWidth())
        self.generationMethod_comboBox.setSizePolicy(sizePolicy)
        self.generationMethod_comboBox.setMaximumSize(QtCore.QSize(190, 50))
        self.generationMethod_comboBox.setObjectName("generationMethod_comboBox")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.generationMethod_comboBox.addItem("")
        self.gridLayout.addWidget(self.generationMethod_comboBox, 1, 1, 1, 2)
        self.y_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.y_lineEdit.sizePolicy().hasHeightForWidth())
        self.y_lineEdit.setSizePolicy(sizePolicy)
        self.y_lineEdit.setObjectName("y_lineEdit")
        self.gridLayout.addWidget(self.y_lineEdit, 2, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 1, 1, 1)
        self.x_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x_lineEdit.sizePolicy().hasHeightForWidth())
        self.x_lineEdit.setSizePolicy(sizePolicy)
        self.x_lineEdit.setObjectName("x_lineEdit")
        self.gridLayout.addWidget(self.x_lineEdit, 3, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 1, 1, 1)
        self.clear_maze_btn = QtWidgets.QPushButton(self.centralwidget)
        self.clear_maze_btn.setObjectName("clear_maze_btn")
        self.gridLayout.addWidget(self.clear_maze_btn, 8, 1, 1, 2)
        self.solutionMethod = QtWidgets.QComboBox(self.centralwidget)
        self.solutionMethod.setObjectName("solutionMethod")
        self.solutionMethod.addItem("")
        self.solutionMethod.addItem("")
        self.solutionMethod.addItem("")
        self.solutionMethod.addItem("")
        self.solutionMethod.addItem("")
        self.gridLayout.addWidget(self.solutionMethod, 9, 1, 1, 2)
        self.convert_to_graph_btn = QtWidgets.QPushButton(self.centralwidget)
        self.convert_to_graph_btn.setObjectName("convert_to_graph_btn")
        self.gridLayout.addWidget(self.convert_to_graph_btn, 5, 1, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menu_3)
        self.menu_4.setObjectName("menu_4")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.about_action = QtWidgets.QAction(MainWindow)
        self.about_action.setObjectName("about_action")
        self.fullscreen_set_action = QtWidgets.QAction(MainWindow)
        self.fullscreen_set_action.setObjectName("fullscreen_set_action")
        self.fullscreen_exit_action = QtWidgets.QAction(MainWindow)
        self.fullscreen_exit_action.setObjectName("fullscreen_exit_action")
        self.menu.addAction(self.action)
        self.menu_2.addAction(self.about_action)
        self.menu_4.addAction(self.fullscreen_set_action)
        self.menu_4.addAction(self.fullscreen_exit_action)
        self.menu_3.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.generationMethod_comboBox, self.y_lineEdit)
        MainWindow.setTabOrder(self.y_lineEdit, self.x_lineEdit)
        MainWindow.setTabOrder(self.x_lineEdit, self.generateBtn)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LabyrinthApp"))
        self.generateBtn.setStatusTip(_translate("MainWindow", "Единичная генерация лабиринта"))
        self.generateBtn.setText(_translate("MainWindow", "Сгенерировать"))
        self.solution_stat_collect_btn.setText(_translate("MainWindow", "Собрать статистику"))
        self.generation_stat_collection_btn.setText(_translate("MainWindow", "Собрать статистику"))
        self.solve_btn.setText(_translate("MainWindow", "Решить"))
        self.generationMethod_comboBox.setStatusTip(_translate("MainWindow", "Способ генерации лабиринта (И - идеальный лабиринт, Н - неидеальный лабиринт)"))
        self.generationMethod_comboBox.setItemText(0, _translate("MainWindow", "Выберите способ генерации"))
        self.generationMethod_comboBox.setItemText(1, _translate("MainWindow", "Растущее дерево (И)"))
        self.generationMethod_comboBox.setItemText(2, _translate("MainWindow", "Олдоса-Бродера (И)"))
        self.generationMethod_comboBox.setItemText(3, _translate("MainWindow", "Уилсона (И)"))
        self.generationMethod_comboBox.setItemText(4, _translate("MainWindow", "Итеративная версия поиска по глубине (И)"))
        self.generationMethod_comboBox.setItemText(5, _translate("MainWindow", "Бинарное дерево (И)"))
        self.generationMethod_comboBox.setItemText(6, _translate("MainWindow", "Эллера (И)"))
        self.generationMethod_comboBox.setItemText(7, _translate("MainWindow", "Крускала (И)"))
        self.generationMethod_comboBox.setItemText(8, _translate("MainWindow", "Прима (И)"))
        self.generationMethod_comboBox.setItemText(9, _translate("MainWindow", "sidewinder (И)"))
        self.generationMethod_comboBox.setItemText(10, _translate("MainWindow", "division (И)"))
        self.generationMethod_comboBox.setItemText(11, _translate("MainWindow", "serpentine (Н)"))
        self.generationMethod_comboBox.setItemText(12, _translate("MainWindow", "small_rooms (Н)"))
        self.generationMethod_comboBox.setItemText(13, _translate("MainWindow", "spiral (Н)"))
        self.y_lineEdit.setStatusTip(_translate("MainWindow", "Размер лабиринта по вертикали"))
        self.y_lineEdit.setText(_translate("MainWindow", "Высота лабиринта"))
        self.label.setText(_translate("MainWindow", "X:"))
        self.x_lineEdit.setStatusTip(_translate("MainWindow", "Размер лабиринта по горизонтали"))
        self.x_lineEdit.setText(_translate("MainWindow", "Ширина лабиринта"))
        self.label_2.setText(_translate("MainWindow", "Y:"))
        self.clear_maze_btn.setText(_translate("MainWindow", "Очистить лабиринт"))
        self.solutionMethod.setStatusTip(_translate("MainWindow", "Способы прохождения лабиринта"))
        self.solutionMethod.setItemText(0, _translate("MainWindow", "Выберите способ решения"))
        self.solutionMethod.setItemText(1, _translate("MainWindow", "ИИ"))
        self.solutionMethod.setItemText(2, _translate("MainWindow", "A-Star"))
        self.solutionMethod.setItemText(3, _translate("MainWindow", "Дейкстра"))
        self.solutionMethod.setItemText(4, _translate("MainWindow", "Поиск по ширине"))
        self.convert_to_graph_btn.setText(_translate("MainWindow", "Преобразовать в граф"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.menu_2.setTitle(_translate("MainWindow", "Справка"))
        self.menu_3.setTitle(_translate("MainWindow", "Вид"))
        self.menu_4.setTitle(_translate("MainWindow", "Полноэкранный режим"))
        self.action.setText(_translate("MainWindow", "Добавить путь"))
        self.about_action.setText(_translate("MainWindow", "О программе"))
        self.fullscreen_set_action.setText(_translate("MainWindow", "Включить"))
        self.fullscreen_exit_action.setText(_translate("MainWindow", "Выключить"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
