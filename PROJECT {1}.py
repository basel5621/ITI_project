from PyQt5 import QtCore, QtWidgets
import pandas as pd
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PyQt5.QtWidgets import QLabel 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Project 1")
        MainWindow.resize(850, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background-color:  #b4a7d6  ;")
        
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 90, 341, 331))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setStyleSheet("background-color: #D3D3D3;")
        

        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 50, 231, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        self.radioButton_Regression = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_Regression.setObjectName("radioButton_Regression")
        self.horizontalLayout_5.addWidget(self.radioButton_Regression)


        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton_Regression)

        self.radioButton_Classification = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_Classification.setObjectName("radioButton_Classification")
        self.horizontalLayout_5.addWidget(self.radioButton_Classification)
        self.buttonGroup.addButton(self.radioButton_Classification)

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
        self.radioButton_Supervised.setGeometry(QtCore.QRect(10, 20, 231, 20))
        self.radioButton_Supervised.setObjectName("radioButton_Supervised")

        
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radioButton_Supervised)

        self.radioButton_Unsupervised = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_Unsupervised.setGeometry(QtCore.QRect(10, 270, 129, 20))
        self.radioButton_Unsupervised.setObjectName("radioButton_Unsupervised")
        self.buttonGroup_2.addButton(self.radioButton_Unsupervised)

        self.radioButton_Supervised.toggled.connect(self.disable_unsupervised)
        self.radioButton_Unsupervised.toggled.connect(self.disable_supervised)

        self.checkBox_k_mean = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_k_mean.setGeometry(QtCore.QRect(20, 300, 129, 20))
        self.checkBox_k_mean.setObjectName("checkBox_k_mean")

     
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 430, 341, 101))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setStyleSheet("background-color: #D3D3D3;")

        
        self.buttonGroup_3 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_3.setObjectName("buttonGroup_3")

        self.radioButton_Confusion_matrix = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_Confusion_matrix.setGeometry(QtCore.QRect(40, 40, 131, 20))
        self.radioButton_Confusion_matrix.setObjectName("radioButton_Confusion_matrix")
        self.buttonGroup_3.addButton(self.radioButton_Confusion_matrix)
        
        self.radioButton_accuracy = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_accuracy.setGeometry(QtCore.QRect(200, 40, 95, 20))
        self.radioButton_accuracy.setObjectName("radioButton_accuracy")
        self.buttonGroup_3.addButton(self.radioButton_accuracy)
        
        self.radioButton_KMeans_Clusters = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_KMeans_Clusters.setGeometry(QtCore.QRect(40, 70, 150, 20))  
        self.radioButton_KMeans_Clusters.setObjectName("radioButton_KMeans_Clusters")
        self.buttonGroup_3.addButton(self.radioButton_KMeans_Clusters)

        self.Run = QtWidgets.QPushButton(self.centralwidget ,clicked = self.run_selected_dataset)
        self.Run.setGeometry(QtCore.QRect(490, 500, 81, 31))
        self.Run.setObjectName("Run")
        self.Run.setStyleSheet("background-color: #32CD32; color: white; font-weight: bold;")

        self.Restart = QtWidgets.QPushButton(self.centralwidget)
        self.Restart.setGeometry(QtCore.QRect(630, 500, 81, 31))
        self.Restart.setObjectName("Restart")
        self.Restart.setStyleSheet("background-color: #FF4500; color: white; font-weight: bold;")
        # Connect the Restart button to the restart_app method
        self.Restart.clicked.connect(self.restart_app)

        self.dataset_combo = QtWidgets.QComboBox(self.centralwidget)
        self.dataset_combo.setGeometry(QtCore.QRect(30, 40, 361, 31))
        self.dataset_combo.setObjectName("dataset_combo")
        self.dataset_combo.setStyleSheet("background-color: #D3D3D3;")
        self.dataset_combo.addItem("")
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
        
        # Error message label (corrected initialization)
        self.error_label = QLabel("", self.centralwidget) 
        self.error_label.setGeometry(QtCore.QRect(30, 540, 500, 30)) 
        self.error_label.setStyleSheet("color: red; font-size: 18px;")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Project 1"))
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
        self.radioButton_KMeans_Clusters.setText(_translate("MainWindow", "kmean_clusters"))
        self.Run.setText(_translate("MainWindow", "Run"))
        self.Restart.setText(_translate("MainWindow", "Restart"))
   
        self.dataset_combo.setItemText(0, _translate("MainWindow", "Select Data"))
        self.dataset_combo.setItemText(1, _translate("MainWindow", "Email Spam Detection"))
        self.dataset_combo.setItemText(2, _translate("MainWindow", "Document clustering "))
        self.dataset_combo.setItemText(3, _translate("MainWindow", "Platform Price Prediction"))


    def run_selected_dataset(self):
        self.error_label.setText('')
        self.valid_algorithm_selected = False
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
            self.error_label.setText("Please select a valid dataset")

        # Plot accuracy or confusion matrix if selected
        if self.valid_algorithm_selected == True:
            
            if self.radioButton_accuracy.isChecked():
                self.plot_accuracy_comparison()
            elif self.radioButton_Confusion_matrix.isChecked():  
                self.show_confusion_matrix()
            elif self.radioButton_KMeans_Clusters.isChecked():
                self.show_k_means_clusters()
            else:
                self.error_label.setText("Please select one of validition options")
            
  ############################# data #########################      


    def if_Email_Spam_selected(self):
        if self.radioButton_Supervised.isChecked():
            if self.radioButton_Regression.isChecked():
                self.error_label.setText("This dataset is not compatible with regression analysis.")
            elif self.radioButton_Classification.isChecked():
                df=pd.read_csv('spam_ham_dataset_cleaned.csv')
                X=df['text_cleaned']
                X=X.fillna('')
                y=df['label_num']        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                TF= TfidfVectorizer()
                X_train_TFIDF= TF.fit_transform(X_train)
                X_test_TFIDF= TF.transform(X_test)
                
                if self.checkBox_Linear_Regression.isChecked():
                    self.error_label.setText("Linear Regression algorithm is not valid for this data.")
                else:
                    if self.checkBox_KNN.isChecked():
                        self.run_knn_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
                        self.valid_algorithm_selected = True
                
                    if self.checkBox_Naive_Bayes.isChecked():
                        self.run_naive_bayes_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
                        self.valid_algorithm_selected = True
                
                    if self.checkBox_Decision_Tree.isChecked():
                        self.run_decision_tree_spam(X_train_TFIDF, y_train, y_test,X_test_TFIDF)
                        self.valid_algorithm_selected = True
                
                    elif self.valid_algorithm_selected == False:
                        self.error_label.setText("No algorithm selected.") 
        
            else:
                self.error_label.setText("Please select Classification  or Regression ")
        elif self.radioButton_Unsupervised.isChecked():
            self.error_label.setText("This dataset is not compatible with Unsupervised algorithm.")
        else:
            self.error_label.setText("Please select Supervised or unSupervised ")
        

            
                
    def if_Price_prediction_selected(self):
        if self.radioButton_Supervised.isChecked():
            if self.radioButton_Regression.isChecked():
                df = pd.read_csv("CarPrice_Assignment.csv")
                # Extract brand and model from CarName
                df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
                df['model'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))
                # Define categorical and numerical columns
                categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
                        
                numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                                     'peakrpm', 'citympg', 'highwaympg']
                # Encoding categorical variables
                label_encoder = LabelEncoder()
                for column in categorical_columns:
                    df[column] = label_encoder.fit_transform(df[column])
                # Feature engineering
                df['power_to_weight_ratio'] = df['horsepower'] / df['curbweight']
                for column in numerical_columns:
                    df[f'{column}_squared'] = df[column] ** 2
                df['log_enginesize'] = np.log(df['enginesize'] + 1)
                # Feature scaling
                scaler = StandardScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
                x= df.drop(['price', 'CarName'], axis=1)  # Include the engineered features and CarName
                y = df['price']
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            
                
                if self.checkBox_Naive_Bayes.isChecked():
                    self.error_label.setText("Naive Bayes algorithm is not valid for this data.") 
                else:
                    if self.checkBox_KNN.isChecked():
                        self.run_knn_regression(x_train, y_train, x_test, y_test)
                        self.valid_algorithm_selected = True
                    
                    
                    if self.checkBox_Linear_Regression.isChecked():
                        self.linear_regression(x_train, y_train, x_test, y_test)
                        self.valid_algorithm_selected = True
                    
                    
                    if self.checkBox_Decision_Tree.isChecked():
                        self.decision_tree(x_train, y_train, x_test, y_test)
                        self.valid_algorithm_selected = True 
                    
                    elif self.valid_algorithm_selected ==False:
                        self.error_label.setText("No algorithm selected.") 
                
            elif self.radioButton_Classification.isChecked():
                self.error_label.setText("This dataset is not compatible with classification analysis.")  
            else:
                self.error_label.setText("Please select regression or classification ")
        
        elif self.radioButton_Unsupervised.isChecked():
            self.error_label.setText("This dataset is not compatible with Unsupervised algorithm.")
        else:
            self.error_label.setText("Please select Supervised or unSupervised ")
                
        
    def if_Document_data_selected(self):
        
        if self.radioButton_Unsupervised.isChecked():
            df = pd.read_csv('file.txt.zip')
            #cleaing
            def get_label(text):
                return int(text[0])
            df['label'] = df['5485'].apply(lambda x: get_label(x))
            df.columns = ['text', 'label']
            df['text'] = df['text'].str[1:]
            df['text'] = df['text'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
            df['text'] = df['text'].apply(lambda x: re.sub(r"((?<=^)|(?<= )).((?=$)|(?= ))", '', x).strip())
            df.drop_duplicates(inplace=True)
            # Feature extraction
            self.vectorizer = TfidfVectorizer(smooth_idf=False)
            self.tfidf = self.vectorizer.fit_transform(df['text'])
            
        
            # Check which algorithms are selected
            if self.checkBox_k_mean.isChecked(): 
                self.run_k_mean()
            elif self.checkBox_KNN.isChecked() or self.checkBox_Naive_Bayes.isChecked() or self.checkBox_Decision_Tree.isChecked() or self.checkBox_Linear_Regression.isChecked():
                self.error_label.setText("This algorithm is not valid for this data.")  
            else:
                self.error_label.setText("No algorithm selected.")  
        
        
        elif self.radioButton_Supervised.isChecked():
            self.error_label.setText("This dataset is not compatible with Supervised algorithm.")
        else: 
            self.error_label.setText("Please select  unSupervised")
        
       

   ############################################## vaildation ######################## 

    def plot_accuracy_comparison(self):
        algorithms = list(self.accuracies.keys())
        scores = list(self.accuracies.values())

        # Clear the canvas before drawing the new plot
        self.canvas.figure.clear()

        # Create a subplot
        ax = self.canvas.figure.add_subplot(111)

        # Plot accuracy comparison
        ax.bar(algorithms, scores, color=['#b4a7d6', '#FF5733', '#33FF57'])
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison between Models')
        ax.set_ylim(0, 1)

        # Refresh the canvas to display the plot
        self.canvas.draw()

    def show_confusion_matrix(self):
        # Clear the canvas before drawing the new plot
        self.canvas.figure.clear()
        count=self.checkBox_KNN.isChecked()+self.checkBox_Naive_Bayes.isChecked()+self.checkBox_Decision_Tree.isChecked()
        # Choose the corresponding confusion matrix
        if count>1:
            self.error_label.setText("Select only one algorithm to generate the Confusion Matrix.")
            return
        elif self.checkBox_KNN.isChecked():
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

    def show_k_means_clusters(self):
        # Ensure K-Means is checked and data is prepared 
        if self.checkBox_k_mean.isChecked():
            self.run_k_mean()  # This runs the K-Means algorithm
        else:
            return
    
    def plot_KMeans_Clusters_Centroids(self, km):#tfidf):
        num_clusters = 5  # Adjust as necessary
        clusters = km.fit_predict(self.tfidf)
        pca = PCA(n_components=2)
        tfidf_reduced = pca.fit_transform(self.tfidf.toarray())
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        scatter = ax.scatter(tfidf_reduced[:, 0], tfidf_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
        centroids = km.cluster_centers_
        centroids_reduced = pca.transform(centroids)
        ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', s=300, marker='x', label='Centroids')
        ax.set_title('KMeans Clusters and Centroids')
        ax.legend()
        self.canvas.draw()
    ########################################## algorithhms #########################3
    
    # KNN Regressor
    def run_knn_regression(self,x_train, y_train, x_test, y_test):
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(x_train, y_train)
        y_predknn = model.predict(x_test)
        accuracy_knn= r2_score(y_test, y_predknn)
        self.accuracies['KNN'] = accuracy_knn
        self.y_predknn=y_predknn
        self.y_test=y_test

    def linear_regression(self,x_train, y_train, x_test, y_test):
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_predlinear = model.predict(x_test)
        accuracy_linear_regression= r2_score(y_test, y_predlinear)
        self.accuracies['linear_regression'] = accuracy_linear_regression
        self.y_predlinear=y_predlinear
        self.y_test=y_test
        
    def decision_tree(self,x_train, y_train, x_test, y_test):
        model = DecisionTreeRegressor()
        model.fit(x_train, y_train)
        y_predDt = model.predict(x_test)
        accuracy_decision_tree = r2_score(y_test, y_predDt)
        self.accuracies['decision_tree'] = accuracy_decision_tree
        self.y_predDt=y_predDt
        self.y_test=y_test

    
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


    def run_k_mean(self):
        num_clusters = 5  # Adjust this based on your data
        km = KMeans(n_clusters=num_clusters)
        km.fit(self.tfidf)
        
        # Plot clusters
        self.plot_KMeans_Clusters_Centroids(km)

    def restart_app(self):
        
        self.clear_supervised()
        self.clear_unsupervised()
        self.reset_disable()

        self.buttonGroup_2.setExclusive(False)
        self.buttonGroup_3.setExclusive(False)

    # Reset radio buttons
        self.radioButton_Supervised.setChecked(False)
        self.radioButton_Unsupervised.setChecked(False)
    

    # Reset radio buttons in Validation group
        self.radioButton_Confusion_matrix.setChecked(False)
        self.radioButton_accuracy.setChecked(False)  # Reset Accuracy button
        self.radioButton_KMeans_Clusters.setChecked(False)
    
        self.buttonGroup_2.setExclusive(True)
        self.buttonGroup_3.setExclusive(True)
    # Reset combo box selection
        self.dataset_combo.setCurrentIndex(0)  # Reset to "Select Data"

    # Clear any displayed error messages
        self.error_label.setText("")

    # Clear the canvas if any plot is displayed
        self.figure.clear()
        self.canvas.draw()

    def clear_supervised(self):
        self.buttonGroup.setExclusive(False)
        self.radioButton_Regression.setChecked(False)
        self.radioButton_Classification.setChecked(False)  
        self.buttonGroup.setExclusive(True)
        # Reset checkboxes
        self.checkBox_Linear_Regression.setChecked(False)
        self.checkBox_KNN.setChecked(False)
        self.checkBox_Decision_Tree.setChecked(False)
        self.checkBox_Naive_Bayes.setChecked(False)
    def clear_unsupervised(self):
        self.checkBox_k_mean.setChecked(False)

    def disable_supervised(self):
        self.clear_supervised()
        
        self.buttonGroup_3.setExclusive(False)
        self.radioButton_accuracy.setChecked(False)
        self.radioButton_Confusion_matrix.setChecked(False)
        self.radioButton_KMeans_Clusters.setChecked(False)
        self.buttonGroup_3.setExclusive(True)
        
        self.radioButton_Regression.setDisabled(True)
        self.radioButton_Classification.setDisabled(True)

        self.checkBox_Linear_Regression.setDisabled(True)
        self.checkBox_KNN.setDisabled(True)
        self.checkBox_Decision_Tree.setDisabled(True)
        self.checkBox_Naive_Bayes.setDisabled(True)
        self.checkBox_k_mean.setDisabled(False)

        self.radioButton_accuracy.setDisabled(True)
        self.radioButton_Confusion_matrix.setDisabled(True)
        self.radioButton_KMeans_Clusters.setDisabled(False)
        
    
    def disable_unsupervised(self):
        self.clear_unsupervised()
        self.radioButton_Regression.setDisabled(False)  
        self.radioButton_Classification.setDisabled(False)  

        self.checkBox_Linear_Regression.setDisabled(False)  
        self.checkBox_KNN.setDisabled(False)  
        self.checkBox_Decision_Tree.setDisabled(False)  
        self.checkBox_Naive_Bayes.setDisabled(False)  
        self.checkBox_k_mean.setDisabled(True)  

        self.radioButton_accuracy.setDisabled(False)  
        self.radioButton_Confusion_matrix.setDisabled(False)  
        self.radioButton_KMeans_Clusters.setDisabled(True)  

    def reset_disable(self):
        self.clear_supervised()
        self.radioButton_Regression.setDisabled(False)
        self.radioButton_Classification.setDisabled(False)

        self.checkBox_Linear_Regression.setDisabled(False)
        self.checkBox_KNN.setDisabled(False)
        self.checkBox_Decision_Tree.setDisabled(False)
        self.checkBox_Naive_Bayes.setDisabled(False)
        self.checkBox_k_mean.setDisabled(False)

        self.radioButton_accuracy.setDisabled(False)
        self.radioButton_Confusion_matrix.setDisabled(False)
        self.radioButton_KMeans_Clusters.setDisabled(False)

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


# In[ ]:




