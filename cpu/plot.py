# File in which the plot function for loss and accuracy is defined.
import matplotlib.pyplot as plt
import csv
import sys
import os

# Read the first argument from the command line and throw an error if it is not provided
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to a model directory as the first argument.")

# Read the path to the directory containing the accuracies.bin and losses.bin files
path = sys.argv[1]

############# HYPER, FUNCTION, ACCURACY AND LOSS DATA #############

# Read the info file
with open(os.path.join(path, "hyperparameters.bin"), mode='r') as paramsfile:
    params = paramsfile.read()

# Open and read the CSV file
accuracy = []
with open(os.path.join(path, "accuracies.csv"), mode='r') as csvacc:
    
    # Create a CSV reader object from the file
    csvreader = csv.reader(csvacc)
    
    # Iterate over each row in the CSV file
    for row in csvreader:

        # Convert each value to float and extend the list 
        accuracy.extend(float(value) for value in row)

# Open and read the loss CSV file
loss = []
with open(os.path.join(path, "losses.csv"), mode='r') as csvloss:

    # Create a CSV reader object from the file 
    csvreader = csv.reader(csvloss)
    
    # Iterate over each row in the CSV file
    for row in csvreader:

        # Convert each value to float and extend the list
        loss.extend(float(value) for value in row)

# Open and read the times file 
times = []
with open(os.path.join(path, "times.csv"), mode='r') as csvtimes:
    
        # Create a CSV reader object from the file 
        csvreader = csv.reader(csvtimes)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
    
            # Convert each value to float and extend the list
            times.extend(float(value) for value in row)

############# X, Y, UNTRAINED Yhat AND TRAINED Yhat DATA #############

# Read the X csv file
X = []
with open(os.path.join(path, "X.csv"), mode='r') as csvX:
        
            # Create a CSV reader object from the file 
            csvreader = csv.reader(csvX)
            
            # Iterate over each row in the CSV file
            for row in csvreader:
        
                # Convert each value to float and extend the list
                X.extend(float(value) for value in row)

# Read the Y csv file
Y = []
with open(os.path.join(path, "Y.csv"), mode='r') as csvY:
    
        # Create a CSV reader object from the file 
        csvreader = csv.reader(csvY)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
    
            # Convert each value to float and extend the list
            Y.extend(float(value) for value in row)

# Read the untrained Yhat csv file
Yhat_untrained = []
with open(os.path.join(path, "untrained_Yhat.csv"), mode='r') as csvYhat_untrained:
    
        # Create a CSV reader object from the file 
        csvreader = csv.reader(csvYhat_untrained)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
    
            # Convert each value to float and extend the list
            Yhat_untrained.extend(float(value) for value in row)

# Read the trained Yhat csv file
Yhat_trained = []
with open(os.path.join(path, "trained_Yhat.csv"), mode='r') as csvYhat_trained:
    
        # Create a CSV reader object from the file 
        csvreader = csv.reader(csvYhat_trained)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
    
            # Convert each value to float and extend the list
            Yhat_trained.extend(float(value) for value in row)


############# PLOTTING LOSS, ACCURACY AND INFO #############

# Build the accuracy plot
plt.plot(accuracy, color='springgreen', linewidth=4)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

# Put the into a shaded box at the bottom right of the plot
plt.text(
    x=0.65,
    y=0.3,
    s=params,
    fontsize=10,
    color='gray',
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.3),
    transform=plt.gca().transAxes)

# Remove corners from the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# put a background grid
plt.grid(color='gray', linestyle='-', linewidth=0.5)

# Save and display the plot
plt.savefig(os.path.join(path, "accuracy_plot.png"))
plt.show()

# Build the loss plot 
plt.plot(loss, color='violet', linewidth=4)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Put the into a shaded box at the top left of the plot
plt.text(
    x=0.1, 
    y=0.95, 
    s=params, 
    fontsize=10, 
    color='gray',
    verticalalignment='top', 
    bbox=dict(facecolor='white', alpha=0.3), 
    transform=plt.gca().transAxes)

# Remove corners from the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# put a background grid
plt.grid(color='gray', linestyle='-', linewidth=0.5)

# Save and display the plot
plt.savefig(os.path.join(path, "loss_plot.png"))
plt.show()

# Build the time plot using a thick line
plt.plot(times, color='darkorange', linewidth=4)
plt.title('Time per Epoch')
plt.ylabel('Time (s)')
plt.xlabel('Epoch')

# We take the total training time by summing the times needed for each epoch
total_training_time = sum(times)

# Calculate the time per iteration
time_per_epoch = total_training_time / len(times)

# Add the total time to the info box
params += f"Time per iteration: {time_per_epoch:.5f} seconds"
params += f"\nTotal Time: {total_training_time:.2f} seconds"

# Put the into a shaded box at the bottom left of the plot
plt.text(
    x=0.1, 
    y=0.95, 
    s=params, 
    fontsize=10, 
    color='gray',
    verticalalignment='top', 
    bbox=dict(facecolor='white', alpha=0.3), 
    transform=plt.gca().transAxes)

# Remove corners from the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# put a background grid
plt.grid(color='gray', linestyle='-', linewidth=0.5)

# Save and display the plot
plt.savefig(os.path.join(path, "time_plot.png"))
plt.show()

############# PLOTTING Y, UNTRAINED Yhat AND TRAINED Yhat #############

# We want to plot the Y, Yhat_untrained and Yhat_trained values against the X values.
# This will give us a visual representation of how well the model is performing.
# We will plot the Y values in blue, the Yhat_untrained values in red and the Yhat_trained values in green.
plt.plot(X, Y, 'salmon', label="Target Function", linewidth=3)
plt.plot(X, Yhat_untrained, 'violet', label='Untrained', linestyle='dashed',linewidth=2)
plt.plot(X, Yhat_trained, 'teal', label='Trained', linestyle='dashed', linewidth=2)
plt.title('Data vs Model Predictions')
plt.ylabel('Value')
plt.xlabel('X')
plt.legend()

# put a background grid
plt.grid(color='gray', linestyle='-', linewidth=0.5)

# Remove corners from the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot
plt.savefig(os.path.join(path, "data_vs_model.png"))

# Display
plt.show()