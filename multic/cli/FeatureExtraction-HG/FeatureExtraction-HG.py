import sys
import os, girder_client
import numpy as np
import pandas as pd
from tiffslide import TiffSlide
from ctk_cli import CLIArgumentParser
import cv2
import csv
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from skimage.color import rgb2gray


sys.path.append("..")


def calculate_color_features(image, compartment_mask):

    feature_values = {}
    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    for compartment in range(3):  # As there are 3 compartments

        compartment_pixels = image[compartment_mask == (compartment + 1)]

        if len(compartment_pixels) > 0:

            # Mean Color

            mean_color = np.mean(compartment_pixels, axis=0)

            for i, channel_value in enumerate(mean_color):

                feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = channel_value

            # Standard Deviation Color

            std_dev_color = np.std(compartment_pixels, axis=0)

            for i, channel_value in enumerate(std_dev_color):

                feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = channel_value

        else:

            # If compartment has no pixels, set values to zero

            for i in range(3):

                feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = 0.0

                feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = 0.0

    return feature_values

 

# Texture Features

def calculate_texture_features(image, compartment_mask):

    feature_values = {}

    texture_feature_names = ['Contrast', 'Homogeneity', 'Correlation', 'Energy']

    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    for compartment in range(3):  # As there are 3 compartments

        compartment_pixels = (compartment_mask == compartment).astype(np.uint8)

        compartment_image = cv2.bitwise_and(image, image, mask=compartment_pixels)

        compartment_image_gray = rgb2gray(compartment_image)

        compartment_image_gray_uint = (compartment_image_gray * 255).astype(np.uint8)

        texture_matrix = graycomatrix(compartment_image_gray_uint, [1], [0], levels=256, symmetric=True, normed=True)
 
        for i, texture_name in enumerate(texture_feature_names):

            texture_feature_value = graycoprops(texture_matrix, texture_name.lower())

            feature_values[f"{texture_name} {compartment_names[compartment]}"] = texture_feature_value[0][0]

    return feature_values

 
 
def calculate_features(image, compartment_mask):

    color_features_dict = calculate_color_features(image, compartment_mask)

    texture_features_dict = calculate_texture_features(image, compartment_mask)

    all_features = {

        #"Distance Transform Features": distance_transform_features_list,

        "Color Features": color_features_dict,

        "Texture Features": texture_features_dict,

        #"Morphological Features": morphological_features_list

    }

    return all_features

 
def main(args):
    # Load the compartment mask and rgb image

    compartment_mask = cv2.imread(args.sub_compartment_mask, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(args.input_file, cv2.IMREAD_COLOR)

    # Assigning custom labels to compartments. For easier calculations

    # 29 - blue = nuclei,

    # 76- red = PAS

    # 149 - green = Luminal space

    compartment_mask[compartment_mask == 29] = 3

    compartment_mask[compartment_mask == 76] = 2

    compartment_mask[compartment_mask == 149] = 1

    compartment_mask[compartment_mask > 3] = 0

    all_features = calculate_features(image, compartment_mask)

    # Print the color features dictionary

    print(all_features)

    # Write the color features to a CSV file

    csv_filename = args.basedir+"/tmp/features_output.csv"
    
    os.makedirs(args.basedir+"/tmp/")

    folder = args.basedir
    
    gc_folder_id = folder.split('/')[-2]
    
    gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)

    gc.setToken(args.girderToken)

    f = open(csv_filename, 'w')
    f.close()

    with open(csv_filename, mode='w', newline='') as csv_file:

        writer = csv.writer(csv_file)

        for feature_name, value in all_features.items():

            writer.writerow([feature_name, ""])  

            if isinstance(value, dict):  

                for sub_feature_name, sub_value in value.items():

                    writer.writerow(["", f"{sub_feature_name}", sub_value])
    
    file_name = csv_filename.split('/')[-1]

    print("Uploading to DSA")
    gc.uploadFileToFolder(gc_folder_id, csv_filename, name = file_name)
    print("Done")






