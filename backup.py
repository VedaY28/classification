import numpy as np


inputNumber = matrix_empty = np.zeros(28, 28)

with open("data/digitdata/trainingimages", "r") as file:
        for i in range(28):  # Iterate through the first 28 lines
            line = file.readline()
            if not line:  # Check if the file ended prematurely
                print(f"Warning: File ended before reading 28 lines. Read {i} lines.")
                break

            # Read exactly 28 characters from the line
            row_data = line[:28]
            

            # Pad the row with empty strings if it's shorter than 28 characters
            if len(row_data) < 28:
                row_data += ' ' * (28 - len(row_data))

            inputNumber[i, :] = list(row_data)  # Save characters to the matrix row

print("The 'inputNumber' matrix:")
print(inputNumber)
print(f"Shape of inputNumber: {inputNumber.shape}")
