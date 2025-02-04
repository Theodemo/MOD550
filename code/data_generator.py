import os
import numpy as np
import matplotlib.pyplot as plt
import csv

class DataGenerator:
    def __init__(self, noise=40, number_points=100, number_of_datasets=1):
        '''
        noise: The standard deviation of the noise added to the data
        number_points: The number of points in each dataset
        number_of_datasets: The number of datasets to generate
        '''

        self.noise = noise
        self.number_points = number_points
        self.number_of_datasets = number_of_datasets

    def generate_linear_with_noise_data2d(self):
        '''
        Generate 2D data with noise from a linear function

        Returns:
        linear_data: A list of lists. Each list contains two lists: x and y values of the data
        '''
        linear_data = []
        slope = np.random.rand(self.number_of_datasets)
        intercept = np.random.rand(self.number_of_datasets)
        
        for i in range(self.number_of_datasets):
            x = np.random.rand(self.number_points) * self.noise
            y = slope[i] * x + intercept[i] + np.random.rand(self.number_points) * self.noise
            linear_data.append([x, y])
        return linear_data

    def generate_random_data2d(self):
        '''
        Generate 2D random data
        
        Returns:
        random_data: A list of lists. Each list contains two lists: x and y values of the data
        '''
        random_data = []
        for i in range(self.number_of_datasets):
            x = np.random.rand(self.number_points) * self.noise
            y = np.random.rand(self.number_points) * self.noise
            random_data.append([x, y])
        return random_data

    def sum_of_two_dataset(self, dataset1, dataset2):
        '''
        Sum two datasets

        dataset1: A list of lists. Each list contains two lists: x and y values of the data
        dataset2: A list of lists. Each list contains two lists: x and y values of the data

        Returns:
        sum_data: A list of lists. Each list contains two lists: x and y values of the data   
        '''
        sum_data = []
        for i in range(self.number_of_datasets):
            x = np.concatenate((dataset1[i][0], dataset2[i][0]))
            y = np.concatenate((dataset1[i][1], dataset2[i][1]))
            sum_data.append([x, y])
        return sum_data

    def save_data(self, data, filename):
        '''
        Save data to a csv file
        
        data: A list of lists. Each list contains two lists: x and y values of the data
        filename: The name of the file to save the data
        '''
        os.makedirs("data", exist_ok=True) 

        with open(f"data/{filename}.csv", "w") as f: 
            f.write("function_id,x,y\n")
            for i in range(self.number_of_datasets):
                for x_val, y_val in zip(data[i][0], data[i][1]):
                    f.write(f"{i},{x_val},{y_val}\n")

    def save_metadata(self, filename, description):
        '''
        Save metadata to a text file
        
        filename: The name of the file to save the metadata
        description: A description of the dataset

        '''

        os.makedirs("data", exist_ok=True) 

        with open(f"data/{filename}.txt", "w") as f:
            f.write(f"Noise: {self.noise}\n")
            f.write(f"Number of points: {self.number_points}\n")
            f.write(f"Number of datasets: {self.number_of_datasets}\n")
            f.write(f"Description: {description}\n")

    def save_image(self, data, title, filename):
        '''
        Save a plot of the data to a png file
        
        data: A list of lists. Each list contains two lists: x and y values of the data
        title: The title of the plot
        filename: The name of the file to save the plot  
        '''
        os.makedirs("data", exist_ok=True) 

        for i in range(self.number_of_datasets):
            plt.scatter(data[i][0], data[i][1], label=f"Function {i}")
        plt.title(title)
        plt.legend()
        plt.savefig(f"data/{filename}.png")  
        plt.close()

    def plot(self, data, title):
        '''
        Plot the data
        
        data: A list of lists. Each list contains two lists: x and y values of the data
        title: The title of the plot
        '''
        for i in range(self.number_of_datasets):
            plt.scatter(data[i][0], data[i][1], label=f"Function {i}")
        plt.title(title)
        plt.legend()
        plt.show()

    def load_data(self, filename):
        '''
        Load data from a csv file
        
        filename: The name of the file to load the data from
        
        Returns:
        loaded_data: A list of lists. Each list contains two lists: x and y values of the data
        '''
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"Error: {filepath} does not exist")
            return None

        loaded_data = {}

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                function_id, x_val, y_val = int(row[0]), float(row[1]), float(row[2])
                if function_id not in loaded_data:
                    loaded_data[function_id] = [[], []]
                loaded_data[function_id][0].append(x_val)
                loaded_data[function_id][1].append(y_val)

        return [loaded_data[i] for i in sorted(loaded_data.keys())] 

if __name__ == '__main__':
    data_gen = DataGenerator(noise=60, number_points=200, number_of_datasets=3)
    
    linear_data = data_gen.generate_linear_with_noise_data2d()
    random_data = data_gen.generate_random_data2d()
    sum_data = data_gen.sum_of_two_dataset(linear_data, random_data)
    
    data_gen.save_data(sum_data, "sum_data")
    data_gen.save_metadata("metadata", "Dataset of linear and random data")
    data_gen.save_image(sum_data, 'Sum of linear and random dataset', 'sum_data')

    # Load and plot data
    loaded_data = data_gen.load_data("sum_data")
    
    if loaded_data:
        data_gen.plot(loaded_data, "Loaded data")
        print("Load successful")
