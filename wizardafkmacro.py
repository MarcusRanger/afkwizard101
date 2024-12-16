from pyautogui import *
import pyautogui
import cv2
import numpy as np
import keyboard
from random import uniform
import time
import pygetwindow as gw
import easyocr
#from PIL import Image

window_title = "Wizard101"
windows = gw.getWindowsWithTitle(window_title)
original_window_width = 1920
original_window_height = 1080
potion_timer = 58
mana_limit = 10
macrostart = time.time()
window = None

for stuff in windows:
    if stuff.title == window_title:
        window = stuff

def use_potion():
    global macrostart
    global window
    window_rect = (window.left, window.top, window.width, window.height)
    relative_pixel = (286, 878)  # Original pixel location
    relative_x = window_rect[0] + relative_pixel[0]
    relative_y = window_rect[1] + relative_pixel[1]

    outSideBattle = pyautogui.pixel(relative_x, relative_y) == (253, 146, 206)  # Adjusted for window
    if outSideBattle:
        macrostart = time.time()
        window.activate()
        pyautogui.hotkey("ctrl", "o", interval=.1)
        pyautogui.sleep(.5)

def match_template(screenshot, template_path, scales=[.8, .9, 1, 1.1, 1.2], threshold=.9):
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
def get_center_from_bounding_box(bbox):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    # Calculate the center
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return int(center_x), int(center_y)

def read_text_with_easyocr(image_path):
    if not image_path.any(): #weird, check type and then validate(?)
        print("Couldn't find image path to read text!")
        return None
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path, mag_ratio=1.2) #adjust magnification if its inconsistent

    text_output = []
    for (bbox, text, prob) in results:
        print(text, prob)
        if text.isdigit():
            text_output.append((get_center_from_bounding_box(bbox), text, prob))
    if not text_output: return None
    print(text_output, "seguramente")
    return text_output

cyan_bgr = (252, 255, 19)

def find_image_on_screen(template_path):
    global window
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    region = (window_rect[0], window_rect[1], window_rect[2], window_rect[3])

    screenshot = np.array(pyautogui.screenshot(region=region))  # Only screenshot the window region
    match = match_template(screenshot, template_path)
    print(f"Current image {match}")
    if match and match['val'] > .95: #need to be 
        print(f"Image found with value {match['val']}")
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        h, w, _ = template.shape
        top_left = match['loc']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        crop_img = screenshot[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        debug_image_path = 'videopictures/cropped_processed_image_found.PNG'
        found_image = cv2.imwrite(debug_image_path, crop_img)
        if found_image:
            return debug_image_path, match['loc']
        print("Error with writing image to folder.")
        return None
    else:
        print("Image not found on screen.")
        return None
    
def read_vitality():
    global window
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    region = (window_rect[0], window_rect[1], window_rect[2], window_rect[3])
    image = np.array(pyautogui.screenshot(region=region))
    height, width, _ = image.shape
    x_start, y_start = 0, int(height * 0.75)  # Start from the bottom-left quarter
    x_end, y_end = int(width * 0.25), height  # Cover 25% width and bottom quarter
    # Crop the region of interest
    cropped_image = image[y_start:y_end, x_start:x_end]
    debug_image_path = 'videopictures/NOWcropped_processed_image_found.PNG'
    cv2.imwrite(debug_image_path, cropped_image)
    
    results = read_text_with_easyocr(cropped_image) #keep in mind this function doesnt have any explicit checks for numbers
    if not results:
        print("No results found")
        return None
    results.sort(key=lambda cord: cord[0][0]) #want the right most bounding box number as mana will always be the right most
    print(results[-1][1], "Soy")
    return results[-1][1] #we only care about mana for now

def in_fight():
    global window
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    
    while not keyboard.is_pressed('q'):
        window.activate()

        with pyautogui.hold('left'):
            pyautogui.sleep(uniform(.7, 1))
        seguramente, location = find_image_on_screen('videopictures/fire_ball.PNG')
        print(seguramente, location)
        if read_vitality() <= mana_limit:
            use_potion()
            #probably use the flee option if use potion doesnt do something.
        secondPersonHere_pixel = (574, 93) #magic numbers for figuring out if an enemy is present
        secondPersonHere_x = window_rect[0] + secondPersonHere_pixel[0]
        secondPersonHere_y = window_rect[1] + secondPersonHere_pixel[1]

        secondPersonHere = pyautogui.pixel(secondPersonHere_x, secondPersonHere_y) == (255, 60, 0) #may have to rewrite this too, no?

        if location and secondPersonHere:
            pyautogui.click(x=location.left, y=location.top, clicks=2, interval=0.25)
            print(location)
            pyautogui.sleep(4)
        elif location and not secondPersonHere:
            pyautogui.moveTo(window_rect[0] + 655, window_rect[1] + 670, 1, pyautogui.easeInQuad) #this is to skip turn I assume
            pyautogui.click(x=window_rect[0] + 655, y=window_rect[1] + 670, clicks=2, interval=0.25)
            pyautogui.sleep(6)


in_fight()