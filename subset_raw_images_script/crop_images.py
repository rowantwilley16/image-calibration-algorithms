import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os 

def create_subset(image_array,dimension):
    # Extract the first 64 rows and columns
    subset_array = image_array[:dimension, :dimension]
    return subset_array

def main(): 

    seed_number = 100
    input_folder_path = r"C:\Users\rowan\Documents\masters\sw_comparison\generated_raw_images\rand_seed_{:d}".format(seed_number)
    save_folder_path = r"C:\Users\rowan\Documents\masters\sw_comparison\subset_raw_images\rand_seed_{:d}".format(seed_number)

    for filename in os.listdir(input_folder_path):
        #check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Open the image
            image_path = os.path.join(input_folder_path, filename)
            image = Image.open(image_path)

            print(f"Image '{filename}' size: {image.size}")

            #convert the image to a numpy array
            image_array = np.array(image)

            #create a subset of the array 
            subset_dimension = 100
            subset_array = create_subset(image_array,subset_dimension)

            #convert the subset array back to an image 

            subset_image = Image.fromarray(subset_array)

            # Save the subset image to a new file
            subset_image.save(os.path.join(save_folder_path, f"subset_{filename}"))

            image.close()
            subset_image.close()

if __name__ == "__main__": 
    main()