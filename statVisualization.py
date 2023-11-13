import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

#region FILES

RNN_559_563_1 = "RNN_559_563_2023.11.04_18.02.xml"
RNN_559_563_2 = "RNN_559_563_2023.11.04_19.51.xml"
RNN_559_563_3 = "RNN_559_563_2023.11.04_20.17.xml"
RNN_559_563_4 = "RNN_559_563_2023.11.04_22.12.xml"
RNN_559_563_5 = "RNN_559_563_2023.11.04_23.25.xml"
RNN_559_563_6 = "RNN_559_563_2023.11.04_23.56.xml"
RNN_559_563_7 = "RNN_559_563_2023.11.05_08.49.xml"
RNN_data1 = "RNN_data1_2023.11.05_12.20.xml"
RNN_data2 = "RNN_data2_2023.11.05_16.22.xml"
RNN_data3_1 = "RNN_data3_2023.11.06_10.17.xml"
RNN_data3_2 = "RNN_data3_2023.11.06_15.26.xml"
RNN_regression_540_1 = "RNN_Regression_540_2023.11.06_18.56.xml"
RNN_regression_540_2 = "RNN_Regression_540_2023.11.06_20.46.xml"
RNN_regression_540_3 = "RNN_Regression_540_2023.11.06_21.28.xml"
RNN_regression_540_4 = "RNN_Regression_540_2023.11.06_21.51.xml"
RNN_regression_540_5 = "RNN_Regression_540_2023.11.06_22.54.xml"
RNN_regression_559_1 = "RNN_Regression_559_2023.11.06_16.36.xml"
RNN_regression_559_2 = "RNN_Regression_559_2023.11.06_17.28.xml"
RNN_regression_559_3 = "RNN_Regression_559_2023.11.06_18.18.xml"
RNN_regression_data1_1 = "RNN_Regression_data1_2023.11.11_11.15.xml"
RNN_regression_data1_2 = "RNN_Regression_data1_2023.11.11_13.02.xml"
RNN_regression_data1_3 = "RNN_Regression_data1_2023.11.11_17.17.xml"
RNN_regression_data3_1 = "RNN_Regression_data3_2023.11.09_20.24.xml"
RNN_regression_data3_2 = "RNN_Regression_data3_2023.11.12_10.41.xml"

All_RNN_classification = [RNN_559_563_1,   \
             RNN_559_563_4,   \
              RNN_data1, RNN_data2, RNN_data3_1, \
             RNN_data3_2]

All_RNN_regression = [ RNN_regression_540_1,  \
              RNN_regression_540_4, RNN_regression_540_5,\
             RNN_regression_559_1,  RNN_regression_559_3, \
             RNN_regression_data1_1, RNN_regression_data1_2, RNN_regression_data1_3, \
             RNN_regression_data3_1, RNN_regression_data3_2]

All_RNN_files = [RNN_559_563_1,   \
             RNN_559_563_4,   \
              RNN_data1, RNN_data2, RNN_data3_1, \
             RNN_data3_2, RNN_regression_540_1,  \
              RNN_regression_540_4, RNN_regression_540_5,\
             RNN_regression_559_1,  RNN_regression_559_3, \
             RNN_regression_data1_1, RNN_regression_data1_2, RNN_regression_data1_3, \
             RNN_regression_data3_1, RNN_regression_data3_2]
#endregion

def process_xml(file_name, include_text):
    # Read and parse the XML file
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Initialize info_text
    info_text = ""

    if include_text:
        # Extract hyperparameters
        info_text = f"File: {file_name}\nHyperparameters:\n"
        hyperparameters = root.find('model/hyperparameters')
        for param in hyperparameters:
            info_text += f"{param.tag}: {param.text.strip()}\n"

        # Extract layers
        info_text += "\nLayers:\n"
        layers = root.find('model/layers')
        for layer in layers:
            info_text += f"{layer.text.strip()}\n"

    # Extract data for plotting
    train_values = []
    prediction_values = []
    for row in root.findall('data/row'):
        try:
            train_val, prediction_val = row.text.split(' train/prediction: ')[1].split('/')
            train_values.append(float(train_val.strip()))
            prediction_values.append(float(prediction_val.strip('[] ')))
        except ValueError as e:
            print(f"Error processing row in {file_name}: {row.text}, error: {e}")

    return train_values, prediction_values, info_text


# User input for displaying text
def single_file_plot(file_name):
    train_values, prediction_values, info_text = process_xml(file_name, True)

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(train_values[0:1440*3], label='Train Values', alpha=0.7)
    plt.plot(prediction_values[0:1440*3], label='Prediction Values', alpha=0.7)
    plt.title(f'Train and Prediction Values for {file_name}')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()

    # Place a text box with the hyperparameters and layers info if requested

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, info_text.strip(), transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

def multiple_files_plot(files):
    include_text = input("Include hyperparameters and layers in the plot? (yes/no): ").strip().lower() == 'yes'
    num_files = len(files)
    fig, axs = plt.subplots(num_files, 1, figsize=(15, 7 * num_files), squeeze=False)

    for i, file_name in enumerate(files):
        train_values, prediction_values, info_text = process_xml(file_name, include_text)

        ax = axs[i, 0] if num_files > 1 else axs[0]

        if train_values and prediction_values:
            ax.plot(train_values[0:1440 * 3], label='Train Values', alpha=0.7)
            ax.plot(prediction_values[0:1440 * 3], label='Prediction Values', alpha=0.7)
            if include_text:
                ax.set_title(f'Train and Prediction Values for {file_name}')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Values')
            ax.legend()

            # Place a text box with the hyperparameters and layers info if requested
            if include_text:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    single_file_plot(RNN_regression_data1_3)