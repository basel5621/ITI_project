from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(850, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 90, 341, 331))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")

        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 50, 231, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        self.radioButton_Regression = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_Regression.setObjectName("radioButton_Regression")
        self.horizontalLayout_5.addWidget(self.radioButton_Regression)

        self.radioButton_Classification = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_Classification.setObjectName("radioButton_Classification")
        self.horizontalLayout_5.addWidget(self.radioButton_Classification)

        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(30, 80, 131, 171))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")

        self.checkBox_Linear_Regression = QtWidgets.QCheckBox(self.verticalLayoutWidget_3)
        self.checkBox_Linear_Regression.setObjectName("checkBox_Linear_Regression")
        self.verticalLayout_7.addWidget(self.checkBox_Linear_Regression)

        self.checkBox_KNN = QtWidgets.QCheckBox(self.verticalLayoutWidget_3)
        self.checkBox_KNN.setObjectName("checkBox_KNN")
        self.verticalLayout_7.addWidget(self.checkBox_KNN)

        self.checkBox_Decision_Tree = QtWidgets.QCheckBox(self.verticalLayoutWidget_3)
        self.checkBox_Decision_Tree.setObjectName("checkBox_Decision_Tree")
        self.verticalLayout_7.addWidget(self.checkBox_Decision_Tree)

        self.checkBox_Naive_Bayes = QtWidgets.QCheckBox(self.verticalLayoutWidget_3)
        self.checkBox_Naive_Bayes.setObjectName("checkBox_Naive_Bayes")
        self.verticalLayout_7.addWidget(self.checkBox_Naive_Bayes)

        self.radioButton_Supervised = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_Supervised.setEnabled(True)
        self.radioButton_Supervised.setGeometry(QtCore.QRect(10, 20, 231, 20))
        self.radioButton_Supervised.setObjectName("radioButton_Supervised")

        self.checkBox_k_mean = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_k_mean.setGeometry(QtCore.QRect(20, 300, 129, 20))
        self.checkBox_k_mean.setObjectName("checkBox_k_mean")

        self.radioButton_Unsupervised = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_Unsupervised.setGeometry(QtCore.QRect(10, 270, 129, 20))
        self.radioButton_Unsupervised.setObjectName("radioButton_Unsupervised")

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 430, 341, 101))
        self.groupBox_2.setObjectName("groupBox_2")

        self.radioButton_Confusion_matrix = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_Confusion_matrix.setGeometry(QtCore.QRect(40, 40, 131, 20))
        self.radioButton_Confusion_matrix.setObjectName("radioButton_Confusion_matrix")
        
        self.radioButton_accuracy = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_accuracy.setGeometry(QtCore.QRect(200, 40, 95, 20))
        self.radioButton_accuracy.setObjectName("radioButton_accuracy")

        self.Run = QtWidgets.QPushButton(self.centralwidget ,clicked = self.run_selected_dataset)
        self.Run.setGeometry(QtCore.QRect(490, 500, 81, 31))
        self.Run.setObjectName("Run")

        self.Restart = QtWidgets.QPushButton(self.centralwidget)
        self.Restart.setGeometry(QtCore.QRect(630, 500, 81, 31))
        self.Restart.setObjectName("Restart")

        self.dataset_combo = QtWidgets.QComboBox(self.centralwidget)
        self.dataset_combo.setGeometry(QtCore.QRect(30, 40, 361, 31))
        self.dataset_combo.setObjectName("dataset_combo")
        self.dataset_combo.addItem("")
        self.dataset_combo.addItem("")
        self.dataset_combo.addItem("")

        self.widget_graph = QtWidgets.QWidget(self.centralwidget)
        self.widget_graph.setGeometry(QtCore.QRect(390, 80, 460, 420))
        self.widget_graph.setObjectName("widget_graph")

        self.graph_layout = QtWidgets.QVBoxLayout(self.widget_graph)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.graph_layout.addWidget(self.canvas)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton_Regression.setText(_translate("MainWindow", "Regression"))
        self.radioButton_Classification.setText(_translate("MainWindow", "Classification"))
        self.checkBox_Linear_Regression.setText(_translate("MainWindow", "Linear Regression"))
        self.checkBox_KNN.setText(_translate("MainWindow", "KNN"))
        self.checkBox_Decision_Tree.setText(_translate("MainWindow", "Decision Tree"))
        self.checkBox_Naive_Bayes.setText(_translate("MainWindow", "Naïve Bayes"))
        self.radioButton_Supervised.setText(_translate("MainWindow", "Supervised"))
        self.checkBox_k_mean.setText(_translate("MainWindow", "K-mean"))
        self.radioButton_Unsupervised.setText(_translate("MainWindow", "Unsupervised"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Validation"))
        self.radioButton_Confusion_matrix.setText(_translate("MainWindow", "Confusion Matrix"))
        self.radioButton_accuracy.setText(_translate("MainWindow", "Accuracy"))
        self.Run.setText(_translate("MainWindow", "Run"))
        self.Restart.setText(_translate("MainWindow", "Restart"))
        self.dataset_combo.setItemText(0, _translate("MainWindow", "Email Spam Detection"))
        self.dataset_combo.setItemText(1, _translate("MainWindow", "document clustering "))
        self.dataset_combo.setItemText(2, _translate("MainWindow", "Platform Price Prediction"))

    def run_selected_dataset(self):
        selected_dataset = self.dataset_combo.currentText()

        # Dictionary to store the selected algorithms and their accuracy
        self.accuracies = {}

        # Running the relevant functions based on dataset selection
        if selected_dataset == "Document clustering ":
            self.if_Document_data_selected()
        elif selected_dataset == "Email Spam Detection":
            self.if_Email_Spam_selected()
        elif selected_dataset == "Platform Price Prediction":
            self.if_Price_prediction_selected()
        else:
            print("Please select a valid dataset")

        # Plot accuracy or confusion matrix if selected
        if self.radioButton_accuracy.isChecked():
            self.plot_accuracy_comparison()
        if self.radioButton_Confusion_matrix.isChecked():
            self.show_confusion_matrix()


    def if_Email_Spam_selected(self):
        df=pd.read_csv('spam_ham_dataset_cleaned.csv')
        X=df['text_cleaned']
        X=X.fillna('')
        y=df['label_num']        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        TF= TfidfVectorizer()
        X_train_TFIDF= TF.fit_transform(X_train)
        X_test_TFIDF= TF.transform(X_test)
        
        if self.checkBox_KNN.isChecked():
            self.run_knn_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
        if self.checkBox_Naive_Bayes.isChecked():
            self.run_naive_bayes_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
        if self.checkBox_Decision_Tree.isChecked():
            self.run_decision_tree_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
    

    def plot_accuracy_comparison(self):
        algorithms = list(self.accuracies.keys())
        scores = list(self.accuracies.values())

        # Clear the canvas before drawing the new plot
        self.canvas.figure.clear()

        # Create a subplot
        ax = self.canvas.figure.add_subplot(111)

        # Plot accuracy comparison
        ax.bar(algorithms, scores, color=['blue', 'orange', 'green'])
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison between Models')
        ax.set_ylim(0, 1)

        # Refresh the canvas to display the plot
        self.canvas.draw()

    def show_confusion_matrix(self):
        # Clear the canvas before drawing the new plot
        self.canvas.figure.clear()

        # Choose the corresponding confusion matrix
        if self.checkBox_KNN.isChecked():
            cm = confusion_matrix(self.y_test, self.y_predknn)
            model_name = "KNN"
        elif self.checkBox_Naive_Bayes.isChecked():
            cm = confusion_matrix(self.y_test, self.y_prednb)
            model_name = "Naive Bayes"
        elif self.checkBox_Decision_Tree.isChecked():
            cm = confusion_matrix(self.y_test, self.y_preddt)
            model_name = "Decision Tree"
        else:
            return

        # Create a subplot on the canvas
        ax = self.canvas.figure.add_subplot(111)

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap=plt.cm.Blues)

        # Set the title and labels for the confusion matrix plot
        ax.set_title(f"Confusion Matrix for {model_name}")

        # Draw the updated plot on the canvas
        self.canvas.draw()


    def run_knn_spam(self, X_vect, y_train, y_test,X_test):
        knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean',weights='distance')
        knn.fit(X_vect, y_train)
        y_predknn = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, y_predknn)
        self.accuracies['KNN'] = knn_accuracy
        self.y_predknn = y_predknn
        self.y_test = y_test

    def run_naive_bayes_spam(self, X_vect, y_train, y_test,X_test):
        nb = MultinomialNB()
        nb.fit(X_vect, y_train)
        y_prednb = nb.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_prednb)
        self.accuracies['Naive Bayes'] = nb_accuracy
        self.y_prednb = y_prednb
        self.y_test = y_test

    def run_decision_tree_spam(self, X_vect, y_train, y_test,X_test):
        dt = DecisionTreeClassifier(criterion= 'gini')
        dt.fit(X_vect, y_train)
        y_preddt = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_preddt)
        self.accuracies['Decision Tree'] = dt_accuracy
        self.y_preddt = y_preddt
        self.y_test = y_test

    def run_knn(self, X_vect, y_train, y_test,X_test):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_vect, y_train)
        y_predknn = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, y_predknn)
        self.accuracies['KNN'] = knn_accuracy
        self.y_predknn = y_predknn
        self.y_test = y_test

    def run_naive_bayes(self, X_vect, y_train, y_test,X_test):
        nb = MultinomialNB()
        nb.fit(X_vect, y_train)
        y_prednb = nb.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_prednb)
        self.accuracies['Naive Bayes'] = nb_accuracy
        self.y_prednb = y_prednb
        self.y_test = y_test

    def run_decision_tree(self, X_vect, y_train, y_test,X_test):
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_vect, y_train)
        y_preddt = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_preddt)
        self.accuracies['Decision Tree'] = dt_accuracy
        self.y_preddt = y_preddt
        self.y_test = y_test

def removeSpecialCharacters(text):
            cleaned_text=re.sub('[^A-Za-z]+', ' ' ,text)
            return cleaned_text.lower()
    
def Removestopwords(text):
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in text.split() if word not in stop_words])

def Stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
