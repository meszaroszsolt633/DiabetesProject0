import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# File name
file_name = 'RNN_data1_2023.11.05_12.20.xml'

# Read and parse the XML file
tree = ET.parse(file_name)
root = tree.getroot()

# Extract hyperparameters
hyperparameters_info = "Hyperparameters:\n"
hyperparameters = root.find('model/hyperparameters')
for param in hyperparameters:
    hyperparameters_info += f"{param.tag}: {param.text.strip()}\n"

# Extract data for plotting
train_values = []
prediction_values = []
for row in root.findall('data/row'):
    # Extract train and prediction values
    try:
        train_val, prediction_val = row.text.split(' train/prediction: ')[1].split('/')
        train_values.append(float(train_val.strip()))
        prediction_values.append(float(prediction_val.strip('[] ')))
    except ValueError as e:
        print(f"Error processing row: {row.text}, error: {e}")

# Check if data_values is empty
if not train_values or not prediction_values:
    print("No data found for plotting.")
else:
    # Plot the data
    plt.figure(figsize=(15, 7))
    plt.plot(train_values[0:1440*3], label='Train Values', alpha=0.7)
    plt.plot(prediction_values[0:1440*3], label='Prediction Values', alpha=0.7)
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Train and Prediction Values with Hyperparameters')
    plt.legend()

    # Place a text box in upper left in axes coords with the hyperparameters info
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, hyperparameters_info.strip(), transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.show()
