# Some functions are written by chatGPT

from config import NUM_OF_STIMULI, METHOD, BLOCK
import re
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random


from config import DATA_DIR, METHOD

def responseCheck(img, response):
    

    match = re.search(r'img_(\d+)\.jpg', img)
    imgNum = int(match.group(1))
    
    if imgNum <= NUM_OF_STIMULI/2:          # if the uimage belongs to an animal
        stimulus = 0   #animal
        if response == 'key_pressed_right':    # if response was animal
            return stimulus, 0, 1    # hit
        elif response == 'key_pressed_left' or response == 'time_out':
            return stimulus, 1, 0    # mis
        else: 
            randChoise = random.choice([0, 1])
            return stimulus, ~randChoise, randChoise
        # else:  raise Exception("the key response does not match the expectation")

    
    if imgNum > NUM_OF_STIMULI/2:          # if the uimage belongs to an object
        stimulus = 1   #object
        if response == 'key_pressed_right' or response == 'time_out':    # if response was animal
            return stimulus, 0, 0    # miss
        elif response == 'key_pressed_left':
            return stimulus, 1, 1    # hit
        else:
            randChoise = random.choice([0, 1])
            return stimulus, randChoise, randChoise
        # else:  raise Exception("the key response does not match the expectation")




def participantAppend(variable, dursSeq, hitsSeq, groundTruthSeq, hitsAnimal, hitsObject,\
                      responseSeq, accuracy, stimuliSeq, sacNumSeq, heatArea):
    variable["dursSeq"].append(dursSeq)
    variable["hitsSeq"].append(hitsSeq)
    variable["groundTruthSeq"].append(groundTruthSeq)
    variable["hitsAnimal"].append(hitsAnimal)
    variable["hitsObject"].append(hitsObject)
    variable["responseSeq"].append(responseSeq)
    variable["accuracy"].append(accuracy)
    variable["stimuliSeq"].append(stimuliSeq)
    variable["sacNumSeq"].append(sacNumSeq)
    variable["heatArea"].append(heatArea)

    return variable



def filterMethodRecs(folders, method, block):
    if method == "GSPV":
        methodNum = "1"
    elif method == "ED":
        methodNum = "2"
    else:  raise Exception("Chosen method in config file is not correct")

    if block != None:
        block = str(block)
        folders = [s for s in folders if s.startswith(methodNum+"_"+block)]
    else: 
        folders = [s for s in folders if s.startswith(methodNum+"_")]
    # folders = [s for s in folders if s.startswith(methodNum+"_")]

    

    return folders

# def filterBlockRecs(folders):


############################################################################################
# The code for this algorithm is extracted from 
# https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms/blob/main/I_VT_Classifier.ipynb
# and redistributed under MIT license.
############################################################################################


# we assume that the frequency is 500Hz so there is 2ms gap between every two samples
def ivt(data,v_threshold):
  Xs = data[:,[0]]
  Ys = data[:,[1]]

  diffX = [] #x values difference
  diffY = [] #y values difference 

  for i in range(len(data) - 1):
    diffX.append(float(Xs[i+1]) - float(Xs[i]) )
    diffY.append(float(Ys[i+1]) - float(Ys[i]) )
  #distance = np.sqrt(np.power(diffX,2) + np.power(diffY,2))
  #velocity = np.divide(distance,2) # 2ms gap!
  #velocity = np.absolute(velocity)
  Velocity = []
  direction=[]
  for i in range(len(diffX)):
    Velocity.append(diffX[i] + diffY[i])
    #direction.append(atan2(diffX[i], diffY[i]))
    velocity=np.divide(Velocity, 2)
    velocity=np.absolute(velocity)

  global mvmts 
  mvmts = []  
  #store 1 in mvmts[] if velocity is less than threshold else store 2
  for v in velocity:
    if(v<v_threshold):
        mvmts.append(0)
    else:
        mvmts.append(1)

  # return mvmts,velocity
  return mvmts


def countSacs_el(lines, start_time, end_time):
    

    # Initialize counters and flags
    ssacc_count = 0
    in_range = False

    # Regular expressions for matching messages and timestamps
    msg_pattern = re.compile(r'^MSG\s+(\d+)\s+(.*)')
    # ssacc_pattern = re.compile(r'^SSACC\s+R\s+(\d+)')
    ssacc_pattern = re.compile(r'^SSACC\s+[RLI]\s+(\d+)')
    timestamp_pattern = re.compile(r'^\d+\s+')

    for line in lines:
        # Check if the line is a message
        msg_match = msg_pattern.match(line)
        if msg_match:
            timestamp = int(msg_match.group(1))
            message = msg_match.group(2)
            if start_time <= timestamp <= end_time:
                in_range = True
            else:
                in_range = False

        # Check if the line is a SSACC message
        ssacc_match = ssacc_pattern.match(line)
        if ssacc_match and in_range:
            ssacc_timestamp = int(ssacc_match.group(1))
            if start_time <= ssacc_timestamp <= end_time:
                ssacc_count += 1
        
        # Check if the line is a timestamp data line
        timestamp_match = timestamp_pattern.match(line)
        if timestamp_match:
            timestamp = int(timestamp_match.group())
            if start_time <= timestamp <= end_time:
                in_range = True
            else:
                in_range = False

    return ssacc_count


def countSacs_lables(preds):
    changes = np.diff(preds)
    starts = np.where(changes == 1)[0]
    return len(starts)


def stimulusAppend(df, imgName, partIndx, sacs, heat, hitOrMiss):
    match = re.search(r'img_(\d+)\.jpg', imgName)
    imgNum = int(match.group(1))
    imgIndx = imgNum-1
    df["hits"][imgIndx] += hitOrMiss
    df["numOfSacs"][imgIndx, partIndx] = sacs 
    df["guasMaps"][imgIndx, partIndx] = heat
    return df

def averageHeatMaps(imageFeats):
    avgMap = np.mean(imageFeats["guasMaps"][:,0:3], axis=1)
    return avgMap


def visualizeHeatMaps(imageFeats, dir, expName, date):
    for i, imageName in enumerate(imageFeats["name"]):
        img = cv2.imread(os.path.join(dir, "images", imageName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        heatmap_normalized = cv2.normalize(imageFeats["avgHeatMaps"][i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_normalized = np.uint8(heatmap_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap_colored, 0.6, img, 0.4, 0)
        cv2.imwrite(os.path.join(dir, "results", date, "heatmaps", expName, imageName), overlay)



def visualizeShadedLinePlot(matrix1, matrix2, title, texty, labels):
    
    # Calculate means and standard deviations for the shaded area
    mean1 = np.mean(matrix1, axis=0)
    std1 = np.std(matrix1, axis=0)
    mean2 = np.mean(matrix2, axis=0)
    std2 = np.std(matrix2, axis=0)
    
    x = np.arange(6, 51)  # X-axis values (1 to 50)
    
    plt.figure(figsize=(12, 6))
    
    # Plot for the first matrix
    plt.plot(x, mean1, label=labels[0], color='blue')
    plt.fill_between(x, mean1 - std1, mean1 + std1, color='blue', alpha=0.3)
    
    # Plot for the second matrix
    plt.plot(x, mean2, label=labels[1], color='red')
    plt.fill_between(x, mean2 - std2, mean2 + std2, color='red', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Trial')
    plt.ylabel(texty)
    plt.legend()
    plt.grid(True)
    plt.show()




def visualizeShadedLinePlot2(matrix1, matrix2, title, texty, labels, std_multiplier=1):
    """
    Visualizes a shaded line plot with means and a shaded area representing standard deviation.
    
    Parameters:
        matrix1 (np.ndarray): First data matrix.
        matrix2 (np.ndarray): Second data matrix.
        title (str): Title of the plot.
        texty (str): Y-axis label.
        labels (list): Labels for the two matrices.
        std_multiplier (float): Multiplier for standard deviation (e.g., 1 for ±1 std, 2 for ±2 std).
    """
    
    # Calculate means and standard deviations for the shaded area
    mean1 = np.mean(matrix1, axis=0)
    std1 = np.std(matrix1, axis=0) * std_multiplier
    mean2 = np.mean(matrix2, axis=0)
    std2 = np.std(matrix2, axis=0) * std_multiplier
    
    x = np.arange(6, 51)  # X-axis values
    
    plt.figure(figsize=(12, 6))
    
    # Plot for the first matrix
    plt.plot(x, mean1, label=labels[0], color='blue')
    plt.fill_between(x, mean1 - std1, mean1 + std1, color='blue', alpha=0.3)
    
    # Plot for the second matrix
    plt.plot(x, mean2, label=labels[1], color='red')
    plt.fill_between(x, mean2 - std2, mean2 + std2, color='red', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Trial')
    plt.ylabel(texty)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualizeParallelPlot(matrix1, matrix2):
    """
    Draws parallel plots for two 10x50 matrices and overlays both plots.
    
    Parameters:
        matrix1 (np.ndarray): The first 10x50 matrix.
        matrix2 (np.ndarray): The second 10x50 matrix.
    """
   
    # Number of variables
    num_vars = matrix1.shape[1]
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Normalize the data to 0-1 range for better visualization
    # def normalize(matrix):
    #     min_val = matrix.min(axis=0)
    #     max_val = matrix.max(axis=0)
    #     return (matrix - min_val) / (max_val - min_val)
    
    # norm_matrix1 = normalize(matrix1)
    # norm_matrix2 = normalize(matrix2)
    
    # Plot for the first matrix
    for row in matrix1:
        plt.plot(range(num_vars), row, color='blue', alpha=0.5, label='GSPV' if row is matrix1[0] else "")
    
    # Plot for the second matrix
    for row in matrix2:
        plt.plot(range(num_vars), row, color='red', alpha=0.5, label='ED' if row is matrix2[0] else "")
    
    # Adding axis labels and legend
    plt.xticks(range(num_vars), [f'Var{i+1}' for i in range(num_vars)], rotation=90)
    plt.xlabel('Variables')
    plt.ylabel('Normalized Value')
    plt.title('Parallel Plot of Two Matrices')
    plt.legend()
    plt.grid(True)
    plt.show()


def visBoxPlot(vector1, vector2, title, labely, colLabels):
    """
    Draws two box plots side by side for the given vectors.
    
    Parameters:
        vector1 (np.ndarray): The first vector.
        vector2 (np.ndarray): The second vector.
    """
    # Ensure the inputs are vectors
    if vector1.ndim != 1 or vector2.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional vectors.")
    
    # Create a figure
    plt.figure(figsize=(6, 6))
    
    # Create box plots
    plt.boxplot([vector1, vector2], colLabels)
    
    # Adding title and labels
    # plt.title('Box Plots achieved accuracies by participants') 
    plt.title(title) 
    plt.xlabel('Representation methods')
    # plt.ylabel('Participants accuracies')
    plt.ylabel(labely)
    
    # Show plot
    plt.show()


# def visViolinPlot(vector1, vector2, title, labely, ledgendTitles):
#     """
#     Draws two box plots side by side for the given vectors.
    
#     Parameters:
#         vector1 (np.ndarray): The first vector.
#         vector2 (np.ndarray): The second vector.
#     """
#     # Ensure the inputs are vectors
#     if vector1.ndim != 1 or vector2.ndim != 1:
#         raise ValueError("Both inputs must be one-dimensional vectors.")
    
#     # Create a figure
#     plt.figure(figsize=(6, 6))
    
#     # Create box plots
#     data = [vector1, vector2]
#     plt.violinplot(data, showmeans=True)

#     # Adding title and labels
#     # plt.title('Box Plots achieved accuracies by participants') 
#     plt.xticks([1, 2], ledgendTitles)
#     plt.title(title) 
#     plt.xlabel('Representation methods')
#     # plt.ylabel('Participants accuracies')
#     plt.ylabel(labely)
    
#     # Show plot
#     plt.show()

def visViolinPlot(vector1_block1, vector1_block2, vector2_block1, vector2_block2, title, labely, ledgendTitles):
    """
    Draws four violin plots side by side for the given vectors.
    
    Parameters:
        vector1_block1 (np.ndarray): Block one of group one.
        vector1_block2 (np.ndarray): Block two of group one.
        vector2_block1 (np.ndarray): Block one of group two.
        vector2_block2 (np.ndarray): Block two of group two.
        title (str): The title of the plot.
        labely (str): The label for the y-axis.
        ledgendTitles (list): The labels for the x-ticks.
    """
    # Ensure the inputs are vectors
    if any(v.ndim != 1 for v in [vector1_block1, vector1_block2, vector2_block1, vector2_block2]):
        raise ValueError("All inputs must be one-dimensional vectors.")
    
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Combine data for the four groups
    data = [vector1_block1, vector1_block2, vector2_block1, vector2_block2]
    
    # Create violin plot with customized colors
    parts = plt.violinplot(data, showmeans=True)
    
    # Set colors for violins (yellow for group one, blue for group two)
    colors = ['yellow', '#add8e6', '#add8e6', 'yellow' ]  # '#add8e6' is light blue
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)  # Adjust transparency if needed

    # Adding title and labels
    plt.xticks([1, 2, 3, 4], ledgendTitles)
    plt.title(title)
    plt.xlabel('Groups and Blocks')
    plt.ylabel(labely)
    
    # Show plot
    plt.show()


def visParallelViolinPlot(vector1_block1, vector1_block2, vector2_block1, vector2_block2, title, labely, ledgendTitles):
    """
    Draws four violin plots side by side for the given vectors, with individual data points
    and lines connecting corresponding points between vector pairs.
    
    Parameters:
        vector1_block1 (np.ndarray): Block one of group one.
        vector1_block2 (np.ndarray): Block two of group one.
        vector2_block1 (np.ndarray): Block one of group two.
        vector2_block2 (np.ndarray): Block two of group two.
        title (str): The title of the plot.
        labely (str): The label for the y-axis.
        ledgendTitles (list): The labels for the x-ticks.
    """
    # Ensure the inputs are vectors
    if any(v.ndim != 1 for v in [vector1_block1, vector1_block2, vector2_block1, vector2_block2]):
        raise ValueError("All inputs must be one-dimensional vectors.")
    
    # Combine data for the four groups
    data = [vector1_block1, vector1_block2, vector2_block1, vector2_block2]
    
    # Check if all vectors have the same length
    n_points = len(vector1_block1)
    if not all(len(v) == n_points for v in data):
        raise ValueError("All vectors must have the same number of elements.")
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create violin plot with customized colors
    parts = plt.violinplot(data, showmeans=True)
    
    # Set colors for violins (yellow for group one, blue for group two)
    colors = ['yellow', '#add8e6', '#add8e6', 'yellow']  # '#add8e6' is light blue
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)  # Adjust transparency if needed

    # Adding title and labels
    plt.xticks([1, 2, 3, 4], ledgendTitles)
    plt.title(title)
    plt.xlabel('Groups and Blocks')
    plt.ylabel(labely)
    
    # Plot individual points and connect corresponding points
    x_positions = [1, 2, 3, 4]
    
    # Scatter points for each group and block
    for i, vec in enumerate(data):
        jitter = np.random.normal(0, 0.05, size=len(vec))  # Add jitter for better visualization
        plt.scatter(np.full(n_points, x_positions[i]) + jitter, vec, color='black', alpha=0.8, zorder=3)
    
    # Draw lines between corresponding points for columns 1->2 and 3->4
    for i in range(n_points):
        # Line between block1 and block2 of group 1 (columns 1 and 2)
        plt.plot([1, 2], [vector1_block1[i], vector1_block2[i]], color='gray', alpha=0.5, zorder=1)
        # Line between block1 and block2 of group 2 (columns 3 and 4)
        plt.plot([3, 4], [vector2_block1[i], vector2_block2[i]], color='gray', alpha=0.5, zorder=1)

    # Show plot
    plt.show()



def visParallelViolinPlot2(vector1_block1, vector1_block2, vector2_block1, vector2_block2, title, labely, ledgendTitles):
    """
    Draws four violin plots side by side for the given vectors, with individual data points
    and lines connecting corresponding points between vector pairs.
    
    Parameters:
        vector1_block1 (np.ndarray): Block one of group one.
        vector1_block2 (np.ndarray): Block two of group one.
        vector2_block1 (np.ndarray): Block one of group two.
        vector2_block2 (np.ndarray): Block two of group two.
        title (str): The title of the plot.
        labely (str): The label for the y-axis.
        ledgendTitles (list): The labels for the x-ticks.
    """
    # Ensure the inputs are vectors
    if any(v.ndim != 1 for v in [vector1_block1, vector1_block2, vector2_block1, vector2_block2]):
        raise ValueError("All inputs must be one-dimensional vectors.")
    
    # Combine data for the four groups
    data = [vector1_block1, vector1_block2, vector2_block1, vector2_block2]
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create violin plot with customized colors
    parts = plt.violinplot(data, showmeans=True)
    
    # Set colors for violins (yellow for group one, blue for group two)
    colors = ['yellow', '#add8e6', '#add8e6', 'yellow']  # '#add8e6' is light blue
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)  # Adjust transparency if needed

    # Adding title and labels
    plt.xticks([1, 2, 3, 4], ledgendTitles, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xlabel('Groups and Blocks', fontsize=12)
    plt.ylabel(labely, fontsize=14)
    
    # Plot individual points and connect corresponding points
    x_positions = [1, 2, 3, 4]
    
    # Scatter points for each group and block with jitter
    for i, vec in enumerate(data):
        jitter = np.random.normal(0, 0.05, size=len(vec))  # Add jitter for better visualization
        plt.scatter(np.full(len(vec), x_positions[i]) + jitter, vec, color='black', alpha=0.8, zorder=3)
    
    # Draw lines between corresponding points within each group where applicable
    min_length_group1 = min(len(vector1_block1), len(vector1_block2))
    for i in range(min_length_group1):
        plt.plot([1, 2], [vector1_block1[i], vector1_block2[i]], color='gray', alpha=0.5, zorder=1)
    
    min_length_group2 = min(len(vector2_block1), len(vector2_block2))
    for i in range(min_length_group2):
        plt.plot([3, 4], [vector2_block1[i], vector2_block2[i]], color='gray', alpha=0.5, zorder=1)

    # Show plot
    plt.show()



def otherBlock(var):
    my_list = ["GSPV", "ED"]

    # Check the content of the variable and extract the other element
    if var == my_list[0]:
        otherblock = my_list[1]
    else:
        otherblock = my_list[0]

    return otherblock


# def visParallelPlot(vector1, vector2, title, labely, trialNames):
#     df = pd.DataFrame({'Trial 1': vector1, 'Trial 2': vector2})

#     # Set the figure size
#     plt.figure(figsize=(6, 6))

#     # Create parallel plot by connecting the two columns with lines
#     for i in range(len(df)):
#         plt.plot([1, 2], [df['Trial 1'][i], df['Trial 2'][i]], marker='o', color='gray')

#     # Add labels and title
#     plt.xticks([1, 2], trialNames)
#     plt.title(title)
#     plt.ylabel(labely)

#     # Show the plot
#     plt.show()



def visParallelPlot(vector1_ED, vector2_ED, vector1_GazeSPV, vector2_GazeSPV, title, labely, trialNames):
    # Create two dataframes for each group
    df_ED = pd.DataFrame({'Trial 1': vector1_ED, 'Trial 2': vector2_ED})
    df_GazeSPV = pd.DataFrame({'Trial 1': vector1_GazeSPV, 'Trial 2': vector2_GazeSPV})

    # Set the figure size
    plt.figure(figsize=(6, 6))

    # Plot the ED group (in blue)
    for i in range(len(df_ED)):
        plt.plot([1, 2], [df_ED['Trial 1'][i], df_ED['Trial 2'][i]], marker='o', color='blue', label="ED -> Gaze-SPV" if i == 0 else "")

    #Plot the Gaze-SPV group (in goldenrod)
    for i in range(len(df_GazeSPV)):
        plt.plot([1, 2], [df_GazeSPV['Trial 1'][i], df_GazeSPV['Trial 2'][i]], marker='o', color='goldenrod', label="Gaze-SPV -> ED" if i == 0 else "")

    # Add labels, title, and legend
    plt.xticks([1, 2], trialNames)
    plt.title(title)
    plt.ylabel(labely)
    plt.legend()

    # Show the plot
    plt.show()


def plot_images_in_2d_space(image_sequence, hits_sequence, response_times, image_folder="images"):
    """
    Plots images in 2D space based on accuracy and response time.

    Parameters:
    - image_sequence: 2D array (list of lists), where each row represents the sequence of image names presented to a participant.
    - hits_sequence: 2D array (list of lists), where each row has 1 or 0 for each image in `image_sequence`, indicating correct (1) or incorrect (0) recognition.
    - response_times: 2D array (list of lists), where each row represents the response time (in ms) for each image in `image_sequence` for each participant.
    - image_folder: Path to the folder containing the images (default is "images").
    """
    # Flatten the data to make calculations easier
    image_list = np.concatenate(image_sequence)
    hits_list = np.concatenate(hits_sequence)
    response_times_list = np.concatenate(response_times)
    
    # Unique images
    unique_images = np.unique(image_list)
    
    # Dictionary to store accuracy and response time for each unique image
    image_accuracy = {}
    image_avg_response_time = {}

    for image in unique_images:
        # Find indices where the current image appears
        indices = np.where(image_list == image)
        
        # Calculate accuracy as mean of hits for the current image
        accuracy = hits_list[indices].mean()
        image_accuracy[image] = accuracy
        
        # Calculate average response time for the current image
        avg_response_time = response_times_list[indices].mean()
        image_avg_response_time[image] = avg_response_time

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    for image in unique_images:
        x = image_accuracy[image]
        y = image_avg_response_time[image]
        
        # Load the image from the specified folder
        image_path = os.path.join(image_folder, image)
        
        try:
            img = Image.open(image_path).convert("RGBA")
            imscatter(x, y, img, ax, zoom=0.02, alpha=0.7)  # Smaller zoom, added transparency
            # Adding markers to verify the positions
            ax.plot(x, y, 'o', markersize=5, color='red')
        except FileNotFoundError:
            print(f"Image {image} not found in {image_folder}. Skipping...")
        
    # Set axis limits with padding
    ax.set_xlim(0.4, 1.1)  # Slight padding around accuracy range (0, 1)
    ax.set_ylim(min(image_avg_response_time.values()) * 0.8, max(image_avg_response_time.values()) * 1.2)

    # Label axes
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Average Response Time (ms)')
    ax.set_title('Image Recognition Scatter Plot')
    plt.grid(True)
    plt.show()

def imscatter(x, y, image, ax=None, zoom=0.15, alpha=1.0):
    """
    Helper function to scatter plot with images.
    
    Parameters:
    - x, y: Coordinates for the image.
    - image: PIL image to be displayed.
    - ax: Matplotlib axis.
    - zoom: Zoom level for image display.
    - alpha: Transparency level for the image.
    """
    if ax is None:
        ax = plt.gca()
    
    # Apply transparency to the image
    image = image.copy()
    image.putalpha(int(alpha * 255))
    
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)