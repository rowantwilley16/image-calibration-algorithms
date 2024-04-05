from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

#global variables 
CAMERA_GAIN = 1

def split_image_into_arrays(path):
    """
    Split an image into separate arrays representing its Red, Green, and Blue color channels.

    This function opens an image located at the specified path and splits it into its three
    color channels: Red, Green, and Blue. It then converts each channel to a separate NumPy array
    of data type uint64.

    Parameters:
        path (str): The file path to the image to be split.

    Returns:
        tuple: A tuple containing arrays representing the Red, Green, and Blue color channels
        of the input image.
            - r_array (numpy.ndarray): Array representing the Red channel.
            - g_array (numpy.ndarray): Array representing the Green channel.
            - b_array (numpy.ndarray): Array representing the Blue channel.

    Global Variables:
        image_width (int): Width of the input image. This variable is updated to reflect the width
        of the Red channel array, which is assumed to be consistent across all color channels.

    Note:
        This function requires the PIL library (imported as 'Image') and the NumPy library (imported as 'np').
    """
    image = Image.open(path)
    #split the image into the 3 color channels and cast it to seperate arrays
    r,g,b       = image.split()
    r_array     = np.array(r,dtype = np.uint64)
    g_array     = np.array(g,dtype = np.uint64)
    b_array     = np.array(b,dtype = np.uint64)

    global image_width
    global image_shape

    image_width = r_array.shape[1]
    image_shape = r_array.shape
    print("Image Shape: ", image_shape)
    return r_array, g_array, b_array

def get_random_num_seed():
    """
    Generate a NumPy random number generator (RNG) with a fixed seed.

    This function sets a fixed seed value (42) for the NumPy random number generator (RNG) to ensure
    reproducibility of random number generation across runs. It then returns the RNG initialized with
    this seed.

    Returns:
        numpy.random.Generator: A NumPy random number generator initialized with the fixed seed (42).

    Notes:
        The fixed seed value (42) ensures that the same sequence of random numbers is generated each time
        this function is called, which is useful for debugging and testing purposes.
    """ 
    #moved to the main block
    #global seed
    #seed = 19
    np.random.seed(seed)
    return np.random.default_rng(seed)
    
def get_read_noise(noise_rng):
    """
        Simulate read noise for each pixel in the input color channels.

        Read noise is simulated by adding Gaussian noise to each pixel in the image. The amount of
        read noise is determined by the READ_NOISE_AMT parameter, which represents the standard deviation
        of the Gaussian distribution. The read noise is scaled based on the camera's gain.

        Parameters:
            r_array (numpy.ndarray): Array representing the Red channel of the image.
            g_array (numpy.ndarray): Array representing the Green channel of the image.
            b_array (numpy.ndarray): Array representing the Blue channel of the image.
            noise_rng (numpy.random.Generator): NumPy random number generator for generating Gaussian noise.

        Returns:
            tuple: A tuple containing arrays representing the read noise for each color channel.
                The tuple contains the following arrays:
                    - read_noise_r: Array of read noise for the Red channel.
                    - read_noise_g: Array of read noise for the Green channel.
                    - read_noise_b: Array of read noise for the Blue channel.
        """ 
    READ_NOISE_AMT  = 20 

    read_noise_r = noise_rng.normal(scale = READ_NOISE_AMT/CAMERA_GAIN, size= image_shape)
    read_noise_g = noise_rng.normal(scale = READ_NOISE_AMT/CAMERA_GAIN, size= image_shape)    
    read_noise_b = noise_rng.normal(scale = READ_NOISE_AMT/CAMERA_GAIN, size= image_shape)

    return read_noise_r, read_noise_g, read_noise_b

def get_dark_current(noise_rng): 
    """
    Simulate dark current noise for each pixel in the input color channels.

    Dark current noise is generated based on a Poisson distribution, modeling the thermal noise
    present in the camera sensor. The dark current is proportional to the exposure time and the
    camera's gain.

    Parameters:
        r_array (numpy.ndarray): Array representing the Red channel of the image.
        g_array (numpy.ndarray): Array representing the Green channel of the image.
        b_array (numpy.ndarray): Array representing the Blue channel of the image.
        noise_rng (numpy.random.Generator): NumPy random number generator for generating Poisson noise.

    Returns:
        tuple: A tuple containing arrays representing the dark current noise for each color channel.
            The tuple contains the following arrays:
                - dark_current_r: Array of dark current noise for the Red channel.
                - dark_current_g: Array of dark current noise for the Green channel.
                - dark_current_b: Array of dark current noise for the Blue channel.
    """
    #Dark Current Noise (per pixel) - Poisson Distribution
    DARK_EXPOSURE                   = 100 
    DARK_CURRENT                    = 0.1

    HOT_PIXELS                      = False 
    HOT_PIXEL_PROBABLILITY          = 0.1 / 100 
    base_current                    = DARK_CURRENT * DARK_EXPOSURE / CAMERA_GAIN

    dark_current_r = noise_rng.poisson(base_current, size = image_shape)
    dark_current_g = noise_rng.poisson(base_current, size = image_shape)
    dark_current_b = noise_rng.poisson(base_current, size = image_shape)

    return dark_current_r, dark_current_g, dark_current_b

def get_col_noise(): 
    """
    Generate random noise parameters for color channels.

    This function generates random values for gain (multiplicative noise) and offset (additive noise)
    for each color channel (Red, Green, Blue) based on normal and uniform distributions.

    Parameters:
        image_width (int): Width of the image, used to determine the number of values to generate for each channel.

    Returns:
        tuple: A tuple containing arrays representing the gain and offset values for each color channel.
            The tuple contains the following arrays:
                - gain_r: Array of Red channel gain values.
                - gain_g: Array of Green channel gain values.
                - gain_b: Array of Blue channel gain values.
                - offset_r: Array of Red channel offset values.
                - offset_g: Array of Green channel offset values.
                - offset_b: Array of Blue channel offset values.
    """
    mean = 1
    std_deviation = 0.05 

    gain_r = np.random.normal(mean, std_deviation, image_width).astype(np.float32)  # Red channel gain values
    gain_g = np.random.normal(mean, std_deviation, image_width).astype(np.float32)  # Green channel gain values
    gain_b = np.random.normal(mean, std_deviation, image_width).astype(np.float32)  # Blue channel gain values

    offset_r = np.random.uniform(0, 10, image_width).astype(np.float32)  # Red channel offset values
    offset_g = np.random.uniform(0, 10, image_width).astype(np.float32)  # Green channel offset values
    offset_b = np.random.uniform(0, 10, image_width).astype(np.float32)  # Blue channel offset values

    return gain_r, gain_g, gain_b, offset_r, offset_g, offset_b

def add_noise(image_array, gain, offset): 
    """
    Adds noise to an image array.

    Parameters:
        image_array (numpy.ndarray): Input image array.
        gain (float): Gain factor to scale the image.
        offset (float): Offset value to add to the image.

    Returns:
        numpy.ndarray: Image array with added noise.
    """
    return (image_array * gain) + offset

def stack_channels_to_image_and_save(r,g,b):
    """
        Stack individual channels (Red, Green, Blue) into an RGB image and save it.

        Parameters:
            r (numpy.ndarray): Array representing the Red channel.
            g (numpy.ndarray): Array representing the Green channel.
            b (numpy.ndarray): Array representing the Blue channel.
            raw_image_path (str): Path to save the generated RGB image.

        Returns:
            None

        """
    altered_r_image = Image.fromarray(r.astype(np.uint8), 'L')
    altered_g_image = Image.fromarray(g.astype(np.uint8), 'L')
    altered_b_image = Image.fromarray(b.astype(np.uint8), 'L')
    rgb_raw_image   =  np.stack((altered_r_image, altered_g_image, altered_b_image), axis=-1)
    raw_image       = Image.fromarray(rgb_raw_image.astype(np.uint8))

    raw_image.save(raw_image_path)
    print("RAW IMAGE GENERATED SUCESSFULLY")

def scale_correction_factors(input_r, input_g, input_b):
    
    fractional_bits = 16 

    scaled_r = input_r * pow(2,fractional_bits)
    scaled_g = input_g * pow(2,fractional_bits)
    scaled_b = input_b * pow(2,fractional_bits)

    return scaled_r, scaled_g, scaled_b

def save_correction_factors(gain, scaled_gain, offset, scaled_offset, inv_gain, character):

    file_name = "rand_seed_{:d}".format(seed)
    np.savetxt(r"C:\Users\rowan\Documents\masters\sw_comparison\correction_factors\{:s}\gain_{:s}.txt".format(file_name,character),             (gain), fmt='%f')
    np.savetxt(r"C:\Users\rowan\Documents\masters\sw_comparison\correction_factors\{:s}\scaled_gain_{:s}.txt".format(file_name, character),      (scaled_gain)   , fmt='%d')
    np.savetxt(r"C:\Users\rowan\Documents\masters\sw_comparison\correction_factors\{:s}\offset_{:s}.txt".format(file_name,character),             (offset)   , fmt='%f')
    np.savetxt(r"C:\Users\rowan\Documents\masters\sw_comparison\correction_factors\{:s}\scaled_offset_{:s}.txt".format(file_name,character),    (scaled_offset)   , fmt='%d')
    np.savetxt(r"C:\Users\rowan\Documents\masters\sw_comparison\correction_factors\{:s}\scaled_inv_gain_{:s}.txt".format(file_name,character),  (inv_gain)   , fmt='%d')
    
def main(): 
    #change the path when a new raw image needs to be generated
    global reference_image_path
    global raw_image_path

    global seed
    seed = 100
    
    ref_img_list = ["ref_grey_0.jpg","ref_grey_50.jpg", "ref_grey_100.jpg","ref_grey_150.jpg","ref_grey_200.jpg", "ref_grey_255.jpg"]
    
    for i in range(len(ref_img_list)):
        reference_image_path    = r"C:\Users\rowan\Documents\masters\sw_comparison\reference_images\{:s}".format(ref_img_list[i])
        raw_image_path          = r"C:\Users\rowan\Documents\masters\sw_comparison\generated_raw_images\rand_seed_{:d}\raw {:s}".format(seed,ref_img_list[i])

        arr_r, arr_g, arr_b = split_image_into_arrays(reference_image_path)

        noise_rng = get_random_num_seed()
        print(noise_rng)
        read_noise_r, read_noise_g, read_noise_b                = get_read_noise(noise_rng)
        dark_curr_r, dark_curr_g, dark_curr_b                   = get_dark_current(noise_rng)

        print("Seed: ",seed)

        gain_r, gain_g, gain_b, offset_r, offset_g, offset_b    = get_col_noise()
    
        print("gain_r:",gain_r)
        scaled_gain_r,scaled_gain_g, scaled_gain_b          = scale_correction_factors(gain_r, gain_g, gain_b)
        scaled_offset_r,scaled_offset_g, scaled_offset_b    = scale_correction_factors(offset_r, offset_g, offset_b)
    
        inv_gain_r,inv_gain_g, inv_gain_b = scale_correction_factors(1/gain_r, 1/gain_g, 1/gain_b)

        save_correction_factors(gain_r,scaled_gain_r, offset_r, scaled_offset_r,inv_gain_r,character='r')
        save_correction_factors(gain_g,scaled_gain_g, offset_g, scaled_offset_g,inv_gain_g,character='g')
        save_correction_factors(gain_b,scaled_gain_b, offset_b, scaled_offset_b,inv_gain_b,character='b')

        #without read noise or dark current
        noisy_r = add_noise(arr_r, gain_r, offset_r)
        noisy_g = add_noise(arr_g, gain_g, offset_g)
        noisy_b = add_noise(arr_b, gain_b, offset_b)

        stack_channels_to_image_and_save(noisy_r, noisy_g, noisy_b)

if __name__ =="__main__": 
    main()
