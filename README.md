Here's an improved and detailed README for your project. I've included the requirements and known limitations based on the script you shared.

---

# AFK Wizard101 Battle Assistant

This is a Python-based automation script designed to assist with battles in **Wizard101**, using OCR (Optical Character Recognition) and AI decision-making. The script automatically identifies card names, enemy health, and game status to choose optimal actions during battle.

---

## **Features**

- Automatically detects and selects the best card based on enemy health.
- Uses template matching to identify key elements in the game window (e.g., buttons like "Draw").
- Incorporates EasyOCR and AI (via `ollama`) for text recognition and decision-making.
- Handles both single-target and area-of-effect (AOE) attacks intelligently.
- Visualizes detected elements (debug bounding boxes for cards and enemies).

---

## **Requirements**

To run this script, you will need:

### **1. Python version**  
Python 3.8+ is recommended. Ensure that Python is installed and available in your system's PATH.

### **2. Required Python packages**
Install the following packages using `pip`:

```bash
pip install pyautogui numpy opencv-python easyocr pygetwindow pillow keyboard rapidfuzz requests
```

| **Package**       | **Description**                                 |
|-------------------|-------------------------------------------------|
| `pyautogui`       | For interacting with the game window (clicks, screenshots, etc.) |
| `numpy`           | For image manipulation and array operations     |
| `opencv-python`   | For template matching and drawing bounding boxes |
| `easyocr`         | For OCR (Optical Character Recognition)         |
| `pygetwindow`     | To get the active game window                   |
| `pillow`          | For image processing (screenshot handling)      |
| `keyboard`        | For keyboard event handling (e.g., stopping the script) |
| `rapidfuzz`       | For fuzzy string matching (card name detection) |
| `requests`        | For AI-based API integration (optional)         |

---

## **How to Run**

1. Make sure **Wizard101** is running and the game window is fully visible.
2. Activate the Python script by executing the following in your terminal or command prompt:
   ```bash
   python main.py
   ```
3. The script will automatically detect when you enter a battle and start assisting with card selection.

Press **Q** to stop the script at any time.

---

## **Current Limitations**

1. **Window Position and Resizing**  
   The script relies on fixed assumptions about the window position and size. If the game window is moved or resized, card detection and clicks may become misaligned. 

   - You may need to tweak the relative positioning values (e.g., in the `click_card` and `get_fight_status` functions).

2. **OCR Accuracy**  
   The OCR may struggle with certain fonts, colors, or low-resolution screenshots. This can cause incorrect or missed detections of card names and enemy health.

   - You can adjust the `preprocess_image_for_ocr()` function to improve detection reliability.
   - Ensure the game is running at a high resolution for better results.

3. **AI Decision-making**  
   The AI may occasionally return incomplete or ambiguous instructions, especially when generating card choices. This requires additional handling for edge cases (e.g., failed parsing of AI responses).

4. **Dependency on Template Matching**  
   The script uses template matching to detect UI elements like the "Draw" button. If the button appearance changes (e.g., due to updates or different screen sizes), you may need to update the template images.

5. **Limited Card Support**  
   The script currently recognizes a predefined list of fire school spells. If you want to expand the script for other schools or custom cards, you will need to update the `fire_spells` dictionary.



## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


Let me know if you'd like any further refinements or additions to the README!
