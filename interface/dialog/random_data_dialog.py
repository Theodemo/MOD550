from PyQt5.QtWidgets import QDialog, QSpinBox, QFormLayout, QPushButton, QBoxLayout

class RandomDataGeneratorDialog(QDialog):
    def __init__(self, parent, current_values):
        super().__init__(parent)
        self.setWindowTitle('Paramètres de génération de données')
        self.setGeometry(200, 200, 400, 200)
        
        self.noise = QSpinBox()
        self.noise.setRange(0, 100)
        self.noise.setValue(current_values['noise'])
        
        self.number_points = QSpinBox()
        self.number_points.setRange(10, 1000)
        self.number_points.setValue(current_values['number_points'])
        
        self.number_of_datasets = QSpinBox()
        self.number_of_datasets.setRange(1, 10)
        self.number_of_datasets.setValue(current_values['number_of_datasets'])
        
        apply_button = QPushButton('Appliquer')
        apply_button.clicked.connect(self.applySettings)
        
        layout = QFormLayout()
        layout.addRow('Niveau de bruit:', self.noise)
        layout.addRow('Nombre de points:', self.number_points)
        layout.addRow('Nombre de jeux de données:', self.number_of_datasets)
        
        btnLine = QBoxLayout(1)
        btnLine.addWidget(apply_button)
        btnLine.addStretch()
        layout.addRow(btnLine)
        
        self.setLayout(layout)
        
    def applySettings(self):
        settings = {
            'noise': self.noise.value(),
            'number_points': self.number_points.value(),
            'number_of_datasets': self.number_of_datasets.value()
        }
        self.parent().update_dataset_settings(settings)
        self.parent().generate_linear_with_noise_data2d()
        
        