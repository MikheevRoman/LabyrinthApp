# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_stats_dlg.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 240)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(500, 240))
        Dialog.setMaximumSize(QtCore.QSize(500, 240))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(73, 148, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 4, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 2)
        self.initial_dimention_le = QtWidgets.QLineEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.initial_dimention_le.sizePolicy().hasHeightForWidth())
        self.initial_dimention_le.setSizePolicy(sizePolicy)
        self.initial_dimention_le.setText("")
        self.initial_dimention_le.setObjectName("initial_dimention_le")
        self.gridLayout.addWidget(self.initial_dimention_le, 0, 3, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(93, 148, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 5, 4, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 2)
        self.finite_dimention_le = QtWidgets.QLineEdit(Dialog)
        self.finite_dimention_le.setText("")
        self.finite_dimention_le.setObjectName("finite_dimention_le")
        self.gridLayout.addWidget(self.finite_dimention_le, 1, 3, 1, 2)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 1, 1, 2)
        self.steps_number_le = QtWidgets.QLineEdit(Dialog)
        self.steps_number_le.setText("")
        self.steps_number_le.setObjectName("steps_number_le")
        self.gridLayout.addWidget(self.steps_number_le, 2, 3, 1, 2)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 1, 1, 2)
        self.repeat_number_le = QtWidgets.QLineEdit(Dialog)
        self.repeat_number_le.setText("")
        self.repeat_number_le.setObjectName("repeat_number_le")
        self.gridLayout.addWidget(self.repeat_number_le, 3, 3, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(73, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 4, 0, 1, 1)
        self.solveMethod_comboBox = QtWidgets.QComboBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.solveMethod_comboBox.sizePolicy().hasHeightForWidth())
        self.solveMethod_comboBox.setSizePolicy(sizePolicy)
        self.solveMethod_comboBox.setMaximumSize(QtCore.QSize(600, 30))
        self.solveMethod_comboBox.setStatusTip("")
        self.solveMethod_comboBox.setObjectName("solveMethod_comboBox")
        self.solveMethod_comboBox.addItem("")
        self.solveMethod_comboBox.addItem("")
        self.solveMethod_comboBox.addItem("")
        self.solveMethod_comboBox.addItem("")
        self.solveMethod_comboBox.addItem("")
        self.gridLayout.addWidget(self.solveMethod_comboBox, 4, 1, 1, 4)
        spacerItem3 = QtWidgets.QSpacerItem(93, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 4, 5, 1, 1)
        self.generationMethod_comboBox = QtWidgets.QComboBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.generationMethod_comboBox.sizePolicy().hasHeightForWidth())
        self.generationMethod_comboBox.setSizePolicy(sizePolicy)
        self.generationMethod_comboBox.setMaximumSize(QtCore.QSize(600, 30))
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
        self.gridLayout.addWidget(self.generationMethod_comboBox, 5, 1, 1, 4)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 6, 0, 1, 2)
        self.collect_stats_btn = QtWidgets.QPushButton(Dialog)
        self.collect_stats_btn.setMaximumSize(QtCore.QSize(150, 16777215))
        self.collect_stats_btn.setObjectName("collect_stats_btn")
        self.gridLayout.addWidget(self.collect_stats_btn, 6, 2, 1, 2)
        spacerItem5 = QtWidgets.QSpacerItem(168, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 6, 4, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Сбор статистики решений"))
        self.label.setText(_translate("Dialog", "Начальная размерность:"))
        self.label_2.setText(_translate("Dialog", "Конечная размерность:"))
        self.label_3.setText(_translate("Dialog", "Количество шагов:"))
        self.label_4.setText(_translate("Dialog", "Количество прогонов:"))
        self.solveMethod_comboBox.setItemText(0, _translate("Dialog", "Выберите способ решения"))
        self.solveMethod_comboBox.setItemText(1, _translate("Dialog", "ИИ"))
        self.solveMethod_comboBox.setItemText(2, _translate("Dialog", "A-Star"))
        self.solveMethod_comboBox.setItemText(3, _translate("Dialog", "Дейкстра"))
        self.solveMethod_comboBox.setItemText(4, _translate("Dialog", "Поиск по ширине"))
        self.generationMethod_comboBox.setStatusTip(_translate("Dialog", "Способ генерации лабиринта (И - идеальный лабиринт, Н - неидеальный лабиринт)"))
        self.generationMethod_comboBox.setItemText(0, _translate("Dialog", "Выберите способ генерации"))
        self.generationMethod_comboBox.setItemText(1, _translate("Dialog", "Растущее дерево (И)"))
        self.generationMethod_comboBox.setItemText(2, _translate("Dialog", "Олдоса-Бродера (И)"))
        self.generationMethod_comboBox.setItemText(3, _translate("Dialog", "Уилсона (И)"))
        self.generationMethod_comboBox.setItemText(4, _translate("Dialog", "Итеративная версия поиска по глубине (И)"))
        self.generationMethod_comboBox.setItemText(5, _translate("Dialog", "Бинарное дерево (И)"))
        self.generationMethod_comboBox.setItemText(6, _translate("Dialog", "Эллера (И)"))
        self.generationMethod_comboBox.setItemText(7, _translate("Dialog", "Крускала (И)"))
        self.generationMethod_comboBox.setItemText(8, _translate("Dialog", "Прима (И)"))
        self.generationMethod_comboBox.setItemText(9, _translate("Dialog", "sidewinder (И)"))
        self.generationMethod_comboBox.setItemText(10, _translate("Dialog", "division (И)"))
        self.generationMethod_comboBox.setItemText(11, _translate("Dialog", "serpentine (Н)"))
        self.generationMethod_comboBox.setItemText(12, _translate("Dialog", "small_rooms (Н)"))
        self.generationMethod_comboBox.setItemText(13, _translate("Dialog", "spiral (Н)"))
        self.collect_stats_btn.setText(_translate("Dialog", "Собрать статистику"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
