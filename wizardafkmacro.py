from pyautogui import *
import pyautogui
import cv2
import numpy as np
import keyboard
from random import uniform
import time
import pygetwindow as gw
import easyocr

window_title = "Wizard101"
windows = gw.getWindowsWithTitle(window_title)
potionTimer = 58
macrostart = time.time()
window =  None
for stuff in windows:
     if stuff.title == window_title:
          window = stuff

    # Take a screenshot of the window



# # Prefered imagebase = 00007FF646C70000
# # "WizardGraphicalClient.exe"+0335DCF8


# baseaddress = 0x7FF646C6FFF0
# staticaddress = 0x0335B518
# pointstaticaddress = c_ulonglong(baseaddress+staticaddress)

# rwm = ReadWriteMemory()
# process = rwm.get_process_by_name("WizardGraphicalClient.exe")
# process.open()
# manapointer = process.get_pointer(pointstaticaddress, offsets=[0x68, 0x0, 0x18, 0x10, 0x10, 0x220, 0x80])

# print(manapointer)
# value = process.read(manapointer)
# print(value)

def timer():
    global macrostart
    global window
    currenttime = time.time()
    outSideBattle = pyautogui.pixel(286,878) == (253,146,206) #if the pet thing isnt present then we are probably in battle
    print((currenttime-macrostart)/60)
    if ((currenttime-macrostart)/60 >= potionTimer) and outSideBattle:
        macrostart = time.time()
        window.activate()
        print("beton iit")
        pyautogui.hotkey("ctrl", "o",interval=.1)
        pyautogui.sleep(.5)

            #idx+=1
# Make sure to install OpenCV: pip install opencv-python
def match_template(screenshot, template_path, scales=[.8,.9,1,1.1,1.2], threshold=.9):
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None:
        raise ValueError("Template image not found or unable to load.")
    
    best_match = {'val': 0, 'loc': (0, 0), 'scale': 0}

    for scale in scales:
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(screenshot_gray, resized_template, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_match['val'] and max_val > threshold:
            best_match = {'val': max_val, 'loc': max_loc, 'scale': scale}
    return best_match


def preprocess_for_ocr(image, target_bgr, threshold_range=0,occlude_top_left_corner=True):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Convert the target BGR color to the HSV color space
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    lower_bound = np.array([target_hsv[0] - threshold_range, 100, 100], dtype=np.uint8)
    upper_bound = np.array([target_hsv[0] + threshold_range, 255, 255], dtype=np.uint8)
    # Create a mask that captures areas of the target color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Isolate the image area with the target color
    isolated = cv2.bitwise_and(image, image, mask=mask)
     # If occlusion of the top left corner is requested
    if occlude_top_left_corner: #
        # Determine the size of the area to occlude
        # These values might need to be adjusted
        height_to_occlude = int(image.shape[0] * 0.2)  # 20% of the height
        width_to_occlude = int(image.shape[1] * 0.2)  # 20% of the width
        cv2.rectangle(isolated, (0, 0), (width_to_occlude, height_to_occlude), (0, 0, 0), -1)
    # Convert to grayscale
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)

    # Increase contrast and apply threshold
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilate and erode to close gaps in text and remove noise
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Resize image to increase the size using interpolation
    resized = cv2.resize(eroded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return resized

def read_text_with_easyocr(image_path):
    # Create a reader object
    reader = easyocr.Reader(['en'])  # 'en' is for English language
    # Read the text from the image
    results = reader.readtext(image_path)

    # Process the results
    text_output = []
    for (bbox, text, prob) in results:
        # bbox is the bounding box information of the text
        # text is the string of text recognized
        # prob is the probability of the prediction
        text_output.append((text, prob))
    if not text_output: return None
    return max(text_output, key=lambda x: x[1])

cyan_bgr = (252, 255, 19)

def read_mana_numbers(template_path):
    screenshot = np.array(pyautogui.screenshot())
    match = match_template(screenshot, template_path)
    
    if match:  
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        h, w, _ = template.shape
        top_left = match['loc']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        crop_img = screenshot[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        debug_image_path = 'videopictures/cropped_processed_image.PNG'
        cv2.imwrite(debug_image_path, crop_img)

        # Assuming preprocess_for_ocr and read_text_with_easyocr are defined
        # Preprocess the cropped image for better OCR
        processed_img = preprocess_for_ocr(crop_img, cyan_bgr)  # BGR for cyan

        cv2.imwrite('videopictures/preprocessed.png', processed_img)
        
        # Use EasyOCR to do OCR on the processed image
        text_from_mana = read_text_with_easyocr('videopictures/preprocessed.png')
        
        return text_from_mana
    
    else:
        print("No match found.")
        return None

def find_image_on_screen(template_path):
    global window
    window.activate()
    screenshot = np.array(pyautogui.screenshot())
    match = match_template(screenshot, template_path)
    
    if match:
        print(f"Image found on with value {match['val']}")
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        h, w, _ = template.shape
        top_left = match['loc']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        crop_img = screenshot[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        debug_image_path = 'videopictures/cropped_processed_image_found.PNG'
        cv2.imwrite(debug_image_path, crop_img)
        # ... (add any additional code for highlighting or actions here)
        return True
    else:
        print("Image not found on screen.")
        return False
#find_image_on_screen('videopictures/fire_ball.PNG')

numbers = read_mana_numbers('videopictures/mana_bar.PNG')

if numbers:
    print(f"Number found on screen: {int(''.join(filter(str.isdigit, numbers[0])))}")
else:
    print("No number found")


def in_fight():

    global window
    window.activate()
    #location = pyautogui.locateOnScreen('videopictures/fire_ball.PNG',grayscale=True, confidence=0.6)
    # Screenshot the area within the game window
    #handles battle logic and plan for when a pig joins the fight late
    location = None
    while keyboard.is_pressed('q') == False:
        timer()
        window.activate()
        with pyautogui.hold('left'):
            pyautogui.sleep(uniform(.7,1))
        window.activate()
        location = pyautogui.locateOnScreen('videopictures/fire_ball.PNG',grayscale=True, confidence=0.4)
        #passbutton = pyautogui.locateOnScreen('videopictures/pass_button.PNG',grayscale=True, confidence=0.6)
        secondPersonHere = pyautogui.pixel(574,93) == (255,60,0) #checks the red background of key logo
        if  location and secondPersonHere:
                pyautogui.click(x=location.left, y= location.top, clicks=2, interval=0.25) 
                print(location)
                pyautogui.sleep(4) #see if both are deadwsws
         
        elif location and not secondPersonHere:
             pyautogui.moveTo(655, 670, 1, pyautogui.easeInQuad)
             pyautogui.click(x=655, y=670, clicks=2, interval=0.25)
             pyautogui.sleep(6)

        