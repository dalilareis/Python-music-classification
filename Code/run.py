import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import design
import process
import singleTest


class Application(QtWidgets.QMainWindow):
    def __init__(self):
        super(Application, self).__init__()

        self.ui = design.Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btn_LDA.clicked.connect(self.test_LDA)
        self.ui.btn_PlotLDA.clicked.connect(self.plot_LDA)
        self.ui.btn_PCA.clicked.connect(self.test_PCA)
        self.ui.btn_PCA_Plot.clicked.connect(self.plot_PCA)
        self.ui.btn_CompareModels.clicked.connect(self.compare_models)

        self.ui.btn_Results.clicked.connect(self.classification)
        self.ui.btn_SaveModel.clicked.connect(self.save_model)
        
        self.ui.btn_Classify.clicked.connect(self.test_music)
        self.ui.btn_OpenFile.clicked.connect(self.get_file)

#-------------------------TAB 1: Pre-Processing_Tests---------------------------------------
   
    #*****************Dimensionality Reduction***********************    
    def test_LDA(self):       
        try:
            selected = self.ui.scaler_Dim.currentText()
            report = process.test_lda(scaler=selected)
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("LDA completed! " + report)
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()

    def plot_LDA(self):       
        try:
            process.plot_lda()
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))

    def test_PCA(self):       
        try:
            scaler = self.ui.scaler_Dim.currentText()
            var = self.ui.PCA_Dim.value()
            report = process.test_pca(scaler=scaler, variance=var)
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("PCA completed!\n" + report)
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()

    def plot_PCA(self):       
        try:
            process.plot_pca()
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))

    #*************Test Classification Models***********************
    
    def compare_models(self):
        try:
            scaler = self.ui.scaler_Dim.currentText()            
            if self.ui.select_PCA.isChecked():
                var = self.ui.PCA_Dim.value()
                file = process.model_testing(scaler=scaler, variance=var, pca=True)
            else:
                file = process.model_testing(scaler=scaler)      
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Model comparison completed! Summary saved to \n" + file)
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()

#--------------------------TAB 2: Model_Results---------------------------------------
   
    def classification(self):
        try:
            model = self.ui.box_Model.currentText() 
            scaler = self.ui.box_Scaler.currentText() 

            if self.ui.btn_combineFeatures.isChecked():
                combo = []
                for item in self.ui.listFeatures.selectedItems():
                    combo.append(item.text())
                setCombo = process.combine_subsets(combo)
                report = process.classify_results(model, scaler=scaler, setX=setCombo)
            else:
                if self.ui.box_Feature.currentText() == "default (all)":
                    report = process.classify_results(model, scaler=scaler)
                else:
                    subset = self.ui.box_Feature.currentText()
                    report = process.classify_results(model, scaler=scaler, subset=subset)
 
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Classification finished! Check the detailed results below...")
            msg.setWindowTitle("Success!")
            msg.setDetailedText(report)
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()

    def save_model(self):
        try:
            model = self.ui.box_Model.currentText() 
            scaler = self.ui.box_Scaler.currentText() 
            result = process.fit_model(model, scaler=scaler) 
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(result + " saved to source as \n final_model.pkl")
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()

#------------------------TAB 3: Classify (new data)---------------------------------------

    def get_file(self):
        music_dir = os.path.expanduser('~/Music')
        filename = QFileDialog.getOpenFileName(self, 'Open file', music_dir,
            "Audio files (*.mp3 *.wav *.wma *.ogg *.flv *.mp4 *.aac)")
        self.ui.text_file.setText(filename[0])

    def test_music(self):
        try:
            filename = self.ui.text_file.text() 
            genre = singleTest.main(filename) 
        except Exception as e:
            QMessageBox.critical(self, "Error!", str(e))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Music classification completed")
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
            self.ui.results.appendPlainText("Genre of " + filename + " is " + genre.upper())
            

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = Application()
    gui.show()
    sys.exit(app.exec_())