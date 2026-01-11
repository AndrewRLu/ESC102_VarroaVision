'''
Detect mites, then display masked results + graph of all past results
'''

import cv2 as cv
import numpy as np
import requests
import time

import csv
import os
import matplotlib.pyplot as plt

#IP webcam res
height = 1080
width = 1920

url = "url" #IP webcam url

def get_image_from_ipcam(url):
    print("Press 'y' to capture, 'q' to quit")
    
    while True:
        # Get current frame
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv.imdecode(img_arr, -1)
        resized_img = cv.resize(img, (1280, 720), interpolation=cv.INTER_AREA)
        
        # Show live preview
        cv.imshow("IP Camera Feed", resized_img)
        
        # Check for key press
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv.destroyAllWindows()
            return None
        elif key == ord('y'):
            # Show the captured frame for confirmation
            print("Press 'y' to SAVE picture, or any other key to TRY AGAIN!...")
            cv.imshow("Confirm Capture - Press 'y' to save, any key to cancel", resized_img)
            confirm_key = cv.waitKey(0) & 0xFF
            
            if confirm_key == ord('y'):
                cv.destroyAllWindows()
                return resized_img
            else:
                # If not confirmed, return to live view
                cv.destroyWindow("Confirm Capture - Press 'y' to save, any key to cancel")

#ref
def reference_photo(url):
    print("1. Take a reference photo:")
    ref = get_image_from_ipcam(url)
    print("Reference photo saved!")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"captured_{timestamp}.png"
    cv.imwrite(image_filename, ref)
    print(f"Image captured and saved as {image_filename}")
    return ref

def get_contours(url):
    print("2. Take a picture of your sticky board:")
    image = get_image_from_ipcam(url) 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"captured_{timestamp}.png"
    cv.imwrite(image_filename, image)
    print(f"Image captured and saved as {image_filename}")
    return image

def calc_ref_area(image):
    '''
    get reference mite size
    '''

    # Normalize the image to the range [0, 255]
    normalized_img = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Convert to HSV color space for better color segmentation
    hsv_image = cv.cvtColor(normalized_img, cv.COLOR_BGR2HSV)

    '''
    HSV ranges for red mites (UPDATE DURING LIVE DEMO BASED ON LIGHTING!!)
    '''

    # lower_red1 = np.array([0, 50, 50])    # Lower range for hue (0-10)
    # upper_red1 = np.array([30, 255, 255])   # Upper range for hue (0-10)
    # lower_red2 = np.array([160, 50, 50])  # Lower range for hue (160-180)
    # upper_red2 = np.array([180, 255, 255])  # Upper range for hue (160-180)

    # lower_red1 = np.array([0, 25, 25])    # Lower range for hue (0-10)
    # upper_red1 = np.array([14, 220, 255])   # Upper range for hue (0-10)
    # lower_red2 = np.array([160, 25, 25])  # Lower range for hue (160-180)
    # upper_red2 = np.array([180, 220, 255])  # Upper range for hue (160-180)  

    
    lower_red1 = np.array([0, 35, 25])    # Lower range for hue (0-10)
    upper_red1 = np.array([14, 220, 255])   # Upper range for hue (0-10)
    lower_red2 = np.array([160, 35, 25])  # Lower range for hue (160-180)
    upper_red2 = np.array([180, 220, 255])  # Upper range for hue (160-180)  


    # Create masks for both red ranges
    mask1 = cv.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        if cv.contourArea(contour) > 5: # tweak this number for reference

            # Create a mask for the current contour
            contour_mask = np.zeros_like(red_mask)
            cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
            
            # Display the contour mask and the original image
            temp = cv.bitwise_and(hsv_image, hsv_image, mask=contour_mask)
            display_image = cv.cvtColor(temp, cv.COLOR_HSV2BGR)
            cv.imshow("good ref?", display_image)
            print("Does this look like a mite?")

            key = cv.waitKey(0)
            if key == ord('y'):
                print("Great, reference mask saved!")
                area = cv.contourArea(contour)
                return area
            
            else:
                print("Let's try again,")

def mite_calculation():

    image = reference_photo(url)
    ref_area = calc_ref_area(image)

    total_area = 0
    total_objects = 0

    # Normalize the image to the range [0, 255]
    normalized_img = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Convert to HSV color space for better color segmentation
    hsv_image = cv.cvtColor(normalized_img, cv.COLOR_BGR2HSV)
    # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    '''
    HSV ranges for red mites (UPDATE DURING LIVE DEMO BASED ON LIGHTING!!)
    '''
    # lower_red1 = np.array([0, 70, 20])      
    # upper_red1 = np.array([30, 255, 150])    
    # lower_red2 = np.array([160, 70, 20])    
    # upper_red2 = np.array([180, 255, 150])

    # lower_red1 = np.array([0, 50, 50])    # Lower range for hue (0-10)
    # upper_red1 = np.array([14, 220, 255])   # Upper range for hue (0-10)
    # lower_red2 = np.array([160, 50, 50])  # Lower range for hue (160-180)
    # upper_red2 = np.array([180, 220, 255])  # Upper range for hue (160-180)   

    
    # lower_red1 = np.array([0, 25, 25])    # Lower range for hue (0-10)
    # upper_red1 = np.array([14, 220, 255])   # Upper range for hue (0-10)
    # lower_red2 = np.array([160, 25, 25])  # Lower range for hue (160-180)
    # upper_red2 = np.array([180, 220, 255])  # Upper range for hue (160-180)  
    
    lower_red1 = np.array([0, 35, 25])    # Lower range for hue (0-10)
    upper_red1 = np.array([14, 220, 255])   # Upper range for hue (0-10)
    lower_red2 = np.array([160, 35, 25])  # Lower range for hue (160-180)
    upper_red2 = np.array([180, 220, 255])  # Upper range for hue (160-180)  

    # Create masks for both red ranges
    mask1 = cv.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv.bitwise_or(mask1, mask2)

    # Find contours in the maskq
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the final result
    final_mask = np.zeros_like(red_mask)

    print(f"ref: {ref_area}")

    for contour in contours:
        area = cv.contourArea(contour)
        print(f"cur area: {area}")
        
        if area > ref_area*100: #filter big things
            # Create a mask for the current contour
            contour_mask = np.zeros_like(red_mask)
            cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
            
            # Display the contour mask and the original image
            temp = cv.bitwise_and(hsv_image, hsv_image, mask=contour_mask)
            display_image = cv.cvtColor(temp, cv.COLOR_HSV2BGR)
            cv.imshow("yay or nay", display_image)
            
            # Prompt the user for input
            print("Is this a mite?")
            key = cv.waitKey(0)  # Wait for user input
            
            # If the user presses 'y', include the object in the final mask
            if key == ord('y'):
                cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)
                total_area += area
                num_objects = area/ref_area
                
                if num_objects > 2.5:
                    total_objects += 1
                else:
                    total_objects += 1
                print("Object included in the final mask.")

            else:
                print("Object excluded from the final mask.")
            
            # Close the contour mask window
            cv.destroyWindow("yay or nay")

        elif area > ref_area*0.3: #filter small particles
            cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)
            total_area += area
            num_objects = area/ref_area
            print("added!")
            if num_objects > 1.8:
                total_objects += 1
            else:
                total_objects += 1
            # print(total_objects)

        else:
            print("removed!")
    
    calc_r = total_objects

    # Display the final results
    mask_f = cv.bitwise_and(hsv_image, hsv_image, mask=final_mask)
    mark_rgb = cv.cvtColor(mask_f, cv.COLOR_HSV2BGR)
    cv.imshow("Mites Mask", mark_rgb)
    cv.imshow("Original", normalized_img)
    print(f"Calculated num: {calc_r:.4f}")

    input_path = r"C:\Users\andre\Desktop\beta_ai\varroa_count_blank.mp4"
    output_path = r"C:\Users\andre\Desktop\beta_ai\varroa_count_res.mp4"
    text = f"{int(total_objects)} mites detected!"
    position = (900, 600)
    add_text_to_video(input_path, output_path, text, position, 3.0, color=(255, 255, 255), thickness=3)
    display_video_on_loop(output_path)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return calc_r

def add_text_to_video(input_path, output_path, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2):
    """
    Add text to each frame of a video and save the result.
    """
    # Open the video file
    cap = cv.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add text to the frame
        cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness, cv.LINE_AA)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()

def display_video_on_loop(video_path):
    """
    Display a video on loop until a key is pressed.
    """
    # Open the video file
    cap = cv.VideoCapture(video_path)
    
    # Create a window
    cv.namedWindow('Final count', cv.WINDOW_NORMAL)
    
    print("Press any key to stop playback...")
    
    while True:
        # Reset video to start
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        # Play the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv.imshow('Video Player', frame)
            
            # Check for key press (wait for 25ms)
            if cv.waitKey(25) != -1:
                cap.release()
                cv.destroyAllWindows()
                return
            
        # If we reached the end, start over unless a key was pressed
        if cv.waitKey(25) != -1:
            break
    
    cap.release()
    cv.destroyAllWindows()

# save in data + plot
def save_data(total_mites, days):
    """Stores the mite count data and determines if an alert should be triggered."""
    mites_per_day = total_mites / days
    alert = mites_per_day > 9
    file_exists = os.path.isfile("mite_data.csv")
    
    with open("mite_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Times Checked", "Total Mites", "Days", "Mites Per Day", "Alert"])
            writer.writerow([1, total_mites, days, mites_per_day, "YES" if alert else "NO"])
        else:
            with open("mite_data.csv", "r") as read_file:
                last_row = list(csv.reader(read_file))[-1]
                last_id = int(last_row[0])
                new_id = last_id + 1
            writer.writerow([new_id, total_mites, days, mites_per_day, "YES" if alert else "NO"])
    
    if alert:
        print(f"⚠️ ALERT: High mite drop rate ({mites_per_day:.2f} mites/day) detected!")
    
    return mites_per_day

def plot_trends():
    """Plots the mite drop trends over time."""
    if not os.path.isfile("mite_data.csv"):
        print("No data available for plotting.")
        return
    
    check_timess, mites_per_day = [], []
    with open("mite_data.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            check_timess.append(row[0])
            mites_per_day.append(float(row[3]))
    
    plt.figure(figsize=(8, 5))
    plt.plot(check_timess, mites_per_day, marker='o', linestyle='-')
    plt.axhline(y=9, color='r', linestyle='--', label='Threshold (9 mites/day)')
    plt.xlabel("Times Checked")
    plt.ylabel("Mites per Day")
    plt.title("Mite Drop Trends")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    days_in_hive = int(input("Enter number of days in hive: "))
    total_mites = 0

    '''
    multiple images (large board) or single image
    '''
    # positions = [
    #     "TOP LEFT", "TOP CENTER", "TOP RIGHT",
    #     "MIDDLE LEFT", "MIDDLE RIGHT",
    #     "BOTTOM LEFT", "BOTTOM CENTER", "BOTTOM RIGHT"
    # ]

    positions = [
    "TOP LEFT"
    ]
    
    for position in positions:
        total_mites += mite_calculation()

    mites_per_day = save_data(total_mites, days_in_hive)

    print(f"Mite drop rate: {mites_per_day:.2f} mites/day")
    plot_trends()