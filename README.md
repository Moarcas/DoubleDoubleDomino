This project contains a Python script for processing Double Double Domino game moves based on provided images. The script uses the OpenCV library for image processing.

## Libraries Required

- OpenCV: version 4.1.1.26
- NumPy: version 1.15.4

You can install these libraries using the following command:

```bash
pip install opencv-python==4.1.1.26 numpy==1.15.4
```


## How to Run

The main functionality of the script is to process game moves from images. Here's how to run it:

1. **Modify Input Paths:**
   - Open the script and modify the `path_read` and `path_write` variables according to the location of your input images and where you want to store the output files.

2. **Run the Script:**
   - Execute the script by running the following command in your terminal:

     ```
     python your_script_name.py
     ```

   Replace `your_script_name.py` with the actual name of your Python script.

3. **Output:**
   - The script processes game moves based on the provided images and generates output files in the specified output directory (`path_write`).

   - Each output file corresponds to a processed image and contains information about the game moves.

## Additional Information

- The script uses various image processing techniques, and you can find detailed information about each step within the code comments.

- Debugging functionality (`processGamesDebug()`) is also available, and you can choose to run it for debugging purposes.

Feel free to reach out if you have any questions or encounter issues while running the script.
