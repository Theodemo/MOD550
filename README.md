# DataGenerator

## Description
DataGenerator is a Python class that allows generating, manipulating, and saving 2D datasets. It enables the creation of noisy linear data and random data, which can then be combined.

## Features
- Generation of noisy linear data
- Generation of random data
- Merging two datasets
- Saving data in CSV format
- Saving metadata in TXT format
- Saving data visualizations as PNG images
- Loading data from a CSV file
- Displaying data as scatter plots

## Installation
No specific installation is required apart from Python and the following libraries:

```bash
pip install numpy matplotlib
```

## Usage
Example usage of the script:

```python
if __name__ == '__main__':
    data_gen = DataGenerator(noise=60, number_points=200, number_of_datasets=3)
    
    linear_data = data_gen.generate_linear_with_noise_data2d()
    random_data = data_gen.generate_random_data2d()
    sum_data = data_gen.sum_of_two_dataset(linear_data, random_data)
    
    data_gen.save_data(sum_data, "sum_data")
    data_gen.save_metadata("metadata", "Dataset of linear and random data")
    data_gen.save_image(sum_data, 'Sum of linear and random dataset', 'sum_data')

    loaded_data = data_gen.load_data("sum_data")
    
    if loaded_data:
        data_gen.plot(loaded_data, "Loaded data")
        print("Load successful")
```

## File Structure
- `data/` : Directory containing generated files
  - `sum_data.csv` : Generated dataset
  - `metadata.txt` : Dataset metadata
  - `sum_data.png` : Scatter plot image

## Author
Developed by Théo de Morais


