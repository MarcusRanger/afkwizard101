import subprocess
from pyautogui import *
import pyautogui
import cv2
import numpy as np
import keyboard
from random import uniform
import time
import pygetwindow as gw
import easyocr
import argparse
from rapidfuzz import fuzz, process
import ollama


# Define the server URL and model parameters
url = "http://localhost:11434/api/generate"
model_name = "deepseek-r1:7b"
#from PIL import Image
window_title = "Wizard101"
windows = gw.getWindowsWithTitle(window_title)
#original_window_width = 1920
#original_window_height = 1080
potion_timer = 58
mana_limit = 10
macrostart = time.time()
window = None
imgs = 0

for stuff in windows:
    if stuff.title == window_title:
        window = stuff

fire_spells = {
    "Fire Cat": ["single", 80],
    "Fire Elf": ["single", 260],  # 50 initial + 210 over 3 rounds
    "Sunbird": ["single", 295],
    "Meteor Strike": ["aoe", 305],
    "Immolate": ["single", 600],  # Deals 200 to self, 600 to target
    "Phoenix": ["single", 515],
    "Helephant": ["single", 625],
    "Fire Dragon": ["aoe", 975],  # 540 initial + 435 over 3 rounds
    "Efreet": ["single", 780],
    "Rain of Fire": ["aoe", 940],  # 130 initial + 810 over 3 rounds
    "Sun Serpent": ["aoe", 785],  # 785-885 to target + 330 to all enemies
    "Raging Bull": ["aoe", 710],  # 710-860 to all enemies
    "Scorching Scimitars": ["aoe", 876],  # Damage divided among targets
    "Scion of Fire": ["single", 1120],  # Deals double if conditions met
    "S'more Machine": ["single", 960],  # 960-1080 to target
    "Blast Off!": ["aoe", 955],  # Delayed damage after 4 rounds
    "Glimpse of Infinity": ["single", 930],  # Two DoTs of 465 over 5 rounds
    "Phantasmania!": ["aoe", 485],
    "A-Baahh-Calypse": ["single", 350],
    "Brimstone Revenant": ["single", 470],
    "Burning Rampage": ["single", 770],  # 70 initial + 700 after 2 rounds
    "Hephaestus": ["single", 425],
    "Krampus": ["single", 305],
    "Nautilus Unleashed": ["single", 380],
    "Whitehart Fire": ["single", 370],
    "Heck Hound": ["single", 130],  # Per pip over 3 rounds
    "Inferno Salamander": ["single", 950],  # Two DoTs of 475 over 5 rounds
    "Link": ["single", 180],  # 30 initial + 150 over 3 rounds
    "Power Link": ["single", 400],  # 55 initial + 345 over 3 rounds
    "Scald": ["aoe", 525],  # Over 3 rounds
    "Detonate": ["single", 0],  # Utility spell, detonates DoTs
    "Backdraft": ["single", 0],  # Utility spell, applies traps
    "Caldera Jinn": ["single", 1440],  # Three DoTs of 480 over 6 rounds
    "Infernal Oni": ["single", 1100],  # Two DoTs of 550 over 5 rounds
    "Bernie Notice": ["single", 1100],  # Two DoTs of 550 over 5 rounds
    "King Artorius (Fire)": ["single", 835],
    "Fire from Above": ["single", 960],
}

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

def preprocess_image_for_ocr(image, scale_factor=1.4):
    # Convert RGB (PyAutoGUI format) to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image with OpenCV
    height, width = image_bgr.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

def read_text_with_easyocr(image):
    # Apply preprocessing before OCR
    processed_image = preprocess_image_for_ocr(image)

    # Save the processed image temporarily (if needed for debugging)
    cv2.imwrite(f'processed_image_for_ocr.png', processed_image)
    # Initialize EasyOCR reader and perform text recognition
    reader = easyocr.Reader(['en'])
    results = reader.readtext(processed_image)

    text_output = []
    for bbox, text, prob in results:
        text_output.append(((get_center_from_bounding_box(bbox)), text, prob))

    return text_output if text_output else None

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

def take_text_from_screen_image():
    global window
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    region = (window_rect[0], window_rect[1], window_rect[2], window_rect[3])
    image = np.array(pyautogui.screenshot(region=region))
    text_from_screenshot = read_text_with_easyocr(image)
    if not text_from_screenshot:
        return None
    return text_from_screenshot

def read_vitality(text_from_screenshot):
    # Define region of interest based on window height and width
    window_rect = (window.left, window.top, window.width, window.height)
    height, width = window_rect[3], window_rect[2]
    
    # Region of interest: bottom-left quarter of the screen
    x_start, y_start = 0, int(height * 0.75)
    x_end, y_end = int(width * 0.25), height

    # Filter text that falls within the region of interest and is numeric
    filtered_numbers = [
        (text, (center_x, center_y))
        for (center_x, center_y), text, _ in text_from_screenshot
        if text.isdigit() and x_start <= center_x <= x_end and y_start <= center_y <= y_end
    ]

    if not filtered_numbers:
        print("No numbers found in the region of interest.")
        return None

    # Sort by x-coordinate to get the rightmost number (e.g., for mana)
    filtered_numbers.sort(key=lambda item: item[1][0])

    # Return the rightmost number as the mana value
    return filtered_numbers[-1][0]

def get_enemy_hp(text_from_screenshot):
    # Define the window region
    window_rect = (window.left, window.top, window.width, window.height)
    window_height = window_rect[3]

    # Define the threshold for height filtering (80% of window height)
    height_threshold = int(window_height * 0.8)

    # Filter for text that contains a "/" and is located above 80% of the window height
    filtered_hp_text = [
        (text, (center_x, center_y))
        for (center_x, center_y), text, _ in text_from_screenshot
        if "/" in text and center_y <= height_threshold
    ]

    return [filtered[0] for filtered in filtered_hp_text]

def get_enemy_position(text_from_screenshot):
    # Define the window region
    window_rect = (window.left, window.top, window.width, window.height)
    window_height = window_rect[3]

    # Define the threshold for height filtering (80% of window height)
    height_threshold = int(window_height * 0.7)

    # Filter for text that contains a "/" and is located above 80% of the window height
    filtered_hp_text = [
        (text, (center_x, center_y))
        for (center_x, center_y), text, _ in text_from_screenshot
        if "rank" in text.lower() and center_y <= height_threshold
    ]

    return [filtered for filtered in filtered_hp_text]

def get_text_in_midsection(text_from_screenshot):
    # Define the window region
    window_rect = (window.left, window.top, window.width, window.height)
    window_height = window_rect[3]

    # Define the midsection range (60% to 63% of the window height)
    mid_start = int(window_height * 0.55)
    mid_end = int(window_height * 0.63)

    # Filter for text that is located within the midsection of the screen
    midsection_text = [
        (text, (center_x, center_y))
        for (center_x, center_y), text, _ in text_from_screenshot
        if mid_start <= center_y <= mid_end
    ]

    return midsection_text

def midsectionOnlyCardNameTest():
    text_from_screenshot = take_text_from_screen_image()
    current_hand = find_best_text_match(get_text_in_midsection(text_from_screenshot))
    print(f"Card names{current_hand}")

def get_fight_status():
    text_from_screenshot = take_text_from_screen_image()
    enemies = get_enemy_hp(text_from_screenshot)
    return len(enemies) > 0

def main():
    global window
    if not window:
        print("No window found; Ending script")
        return
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    
    while not keyboard.is_pressed('q'):
        window.activate()

        in_fight = get_fight_status()
        if  in_fight:
            context_engine()

        with pyautogui.hold('left'):
            pyautogui.sleep(uniform(.7, 1))
        if read_vitality() <= mana_limit:
            use_potion()
        
# Function to send a prompt to the model
messages = []

def prepare_ai_thinking():
    #Generate responses for the ai to cache(greatly speeds up future decision making)
    enemy_health_filter = ["282/282","95/95"] #"282/282", 

    enemy_health_filter2 = ["90/90"]
    #aoe = 1
    #single attack = 2
    cards = {"meteor strike":["aoe",300], "fire cat":["single target",90]}

    cards2 = {"fire cat":["single target",90]}
    #pipe into deepseek r1 with prompt
    context_prompt = f"You are playing a turned base game where you have to pick the best move given the current situation. You can only use one card per turn.\
    The goal is to kill the enemies in the most efficient way possible i.e end the battle as fast as possible. You are given the ability to use two types of attacks.\
    One is single target and another is an area of effect(aoe) attack i.e it hits all enemies. Given the information of the enemies hp bars and the information of the cards,\
    pick the best choice of action. When stating the choice of action either state 'aoe card_name' or 'card_name on enemy x' card_name\
    being the card you picked and the x being which enemy you chose also encase the choice within curly brackets to make parsing your answer easy. The information of the battle will be given to you in the form of a python data table/dictionary\
    label the enemies from 1 up to the length in the data list {enemy_health_filter} the health is a string formatted as 'current_health/base_health' \
    and here are the cards available {cards} for this data the key is the card name and the value is a table with spell type, damage. As a reminder\
        make sure your answer is encased within curly brackets to make parsing your answer easy for code."

    after_context_prompt = f"Given the state of enemies {enemy_health_filter2} and cards {cards2}. Whats the move? Only pick one card."
    
    response = get_model_response(context_prompt)
    print(response)
    response = get_model_response(after_context_prompt)
    print(response)

def get_model_response(prompt):
    # Append the user's message to the messages list
    messages.append({'role': 'user', 'content': prompt})

    # Generate a response from the model
    try:
        response = ollama.chat(model=model_name, messages=messages)
    except Exception as e:
        print(f"Failed to generate response for prompt '{prompt}': {e}")
    # Append the assistant's response to the messages list
    if not response: return "Failed"
    messages.append({'role': 'assistant', 'content': response['message']['content']})
    return response['message']['content']
valid_text_options = fire_spells.keys()

def find_best_text_match(ocr_text):
    # Use rapidfuzz to find the best match for each detected text
    confidence = 50
    matches = {}
    for text, (positionX,positionY) in ocr_text:
        seq, best_match, score = process.extractOne(text, valid_text_options, scorer=fuzz.ratio)
        print(seq, best_match, score, text)
        if best_match >= confidence and (not 'to select' in text.lower() or not 'to discard' in text.lower()): 
            if seq not in matches:
                matches[seq] = []
            matches[seq].append((positionX,positionY))
    return matches

def context_engine():
    global window
    window.activate()
    #grab information on the battle
    text_from_screenshot = take_text_from_screen_image()
    
    #mana = read_vitality(text_from_screenshot)
    enemy_health = get_enemy_hp(text_from_screenshot) # Data formatted for the ai to read
    enemy_position = get_enemy_position(text_from_screenshot) # Data for pyautogui
    current_hand = find_best_text_match(get_text_in_midsection(text_from_screenshot)) # easyOCR has a hard time figuring out card names
    #print(f"The current mana in battle {mana}")
    card_info = {key:fire_spells[key] for key in current_hand.keys()} # Data formatted for the ai to read
    new_current_hand = {key.lower():value for key,value in current_hand.items()} #weird situation with ai not returning properly cased card names
    print(f"The current card info of hand is {card_info}")
    print(f"The current hp of enemies {enemy_health}")
    print(f"The current position of the enemies are {enemy_position}")
    print(f"Text in midsection of screen{current_hand}")
    if not enemy_health or not current_hand:
        print("Not in a battle or error with parsing state of battle")
        return
    #take information from piped and act on it
    after_context_prompt = f"Given the state of enemies {enemy_health} and cards {card_info}. Whats the move? Only pick one card.\
        As a reminder make sure your answer is encased within curly brackets to make parsing your answer easy for code."
    response = get_model_response(after_context_prompt)
    
    if not response:
        print("Couldn't generate game plan")
        return
    #print(response)
    # Deepseek tends to forget that it needs to encase the answer will need a more comprehensive repsonse finding solution
    result = response.split("</think>", 1)
    if not result:
        print("Couldnt find think tag within response")
        return
    if len(result) < 1: 
        print("Result of answer couldnt be parsed")
    result = result[1].strip()
    print(result, "Read alert")
    for key in new_current_hand.keys():
        if key in result.lower():
            click_card(key, new_current_hand)
            return
        
    print("If we reached here then this stupid llm couldnt do the one thing it needed to do")

def click_card(chosen_card, card_pos):
    global window
    if not window:
        print("Couldnt find window")
        return
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    if chosen_card not in card_pos:
        print("Failed to find card to click")
        return
    posX,posY = card_pos[chosen_card][0]
    print("going to click!",chosen_card, posX, posY)
    pyautogui.moveTo(window_rect[0] + posX, window_rect[1] + posY, 1, pyautogui.easeInQuad) 
    pyautogui.click(x=None, y=None, clicks=2, interval=0.25)
    pyautogui.sleep(5)

def readTextInPicture(): # test function for hovering over enemy.
    global window
    window.activate()
    window_rect = (window.left, window.top, window.width, window.height)
    offset_ratio = 0.09  #9% of the window height
    mouse_position_offset = int(window_rect[3] * offset_ratio)  # Scale offset dynamically
    region = (window_rect[0], window_rect[1], window_rect[2], window_rect[3])
    #print()#context engine for picking a card goes here
    screenshot = np.array(pyautogui.screenshot(region=region))
    text_from_screenshot = read_text_with_easyocr(screenshot)
    
    enemy_filter = [item for item in text_from_screenshot if "rank" in item[1].lower()]
    
    #enemy_filter = list(enemy_filter)
    print(enemy_filter)
    for item in enemy_filter:
        pos, text, probablility = item
        posX,posY = pos[0],pos[1]+mouse_position_offset #
        pyautogui.moveTo(window_rect[0] + posX, window_rect[1] + posY, 1, pyautogui.easeInQuad) 
        pyautogui.click(x=None, y=None, clicks=2, interval=0.25)
    

if __name__ == "__main__":
    prepare_ai_thinking()
    main()
    #readTextInPicture()
