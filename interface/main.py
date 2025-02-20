import sys
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QGridLayout, QWidget
from PyQt5.QtGui import QPixmap


from dialog.random_data_dialog import RandomDataGeneratorDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(150, 50, 1920,1080)
        self.dataset_settings = {
            'noise': 40,
            'number_points': 100,
            'number_of_datasets': 1
        }
        self.initUI()
        self.initMenu()
        

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.grid_layout = QGridLayout()
        self.central_widget.setLayout(self.grid_layout)
               
        self.labels = []

        for i in range(2):
            ligne = []  
            for j in range(2):
                label = QLabel(self)  
                ligne.append(label)  
            self.labels.append(ligne)  

        for i in range(2):
            for j in range(2):
                self.grid_layout.addWidget(self.labels[i][j], i, j)
                self.labels[i][j].setScaledContents(True) 
                
                
    def initMenu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        
        generate_random_data_action = QAction('Generate Random Data', self)
        generate_random_data_action.triggered.connect(self.show_random_data_dialog)
        edit_menu.addAction(generate_random_data_action)
        
        
        
    def update_dataset_settings(self, new_settings):
        self.dataset_settings = new_settings
        
    
    def show_random_data_dialog(self):
        dialog = RandomDataGeneratorDialog(self, self.dataset_settings)
        dialog.exec_()
        

    def generate_linear_with_noise_data2d(self):
        data = []
        for i in range(self.dataset_settings['number_of_datasets']):
            x = np.random.rand(self.dataset_settings['number_points'])
            y = 2 * x + 1 + np.random.normal(0, self.dataset_settings['noise'], x.shape)
            data.append((x, y))
        self.display_data(data, 0, 0)
           
        return
    
    def display_data(self, data, row, col):
        for i in range(self.dataset_settings['number_of_datasets']):
            x, y = data[i]
            self.labels[row][col].clear()

            # Create figure and axes
            fig, ax = plt.subplots()

            # Use Seaborn to create the plot
            sns.scatterplot(x=x, y=y, ax=ax)

            # Set the title using Seaborn styling
            ax.set_title(f'Dataset {i+1}')

            # Save the plot to a file in a folder
            save_path = os.path.join("plots", f"dataset_{i+1}.png")
            os.makedirs("plots", exist_ok=True)  # Create the "plots" folder if it doesn't exist
            fig.savefig(save_path)

            # Load the saved image as a QPixmap
            pixmap = QPixmap(save_path)
            self.labels[row][col].setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()