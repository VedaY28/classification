import numpy as np
import samples

"""
def read_k_blocks(filename="trainingimages", k=1):

    blocks = []
    with open("data/digitdata/trainingimages", "r") as file:
            for block_num in range(k):
                block_data = []
                for _ in range(28):
                    line = file.readline()
                    if not line:
                        print(f"Warning: End of file reached before reading all {k} blocks.")
                        return blocks

                    # Read exactly 28 characters, padding with spaces if needed
                    row_data = line[:28].ljust(28)
                    block_data.extend(list(row_data))

                if len(block_data) == 784:
                    matrix = np.array(block_data, dtype='U1').reshape(784, 1)
                    blocks.append(matrix)
                else:
                    print(f"Warning: Could not read a full 28x28 block for block {block_num + 1}.")
                    return blocks
            return blocks

if __name__ == "__main__": 
    num_blocks_to_read = 100000  # Example: Read 3 blocks
    # Note to myself there are 5000 blocks in the training data
    image_blocks = read_k_blocks(k=num_blocks_to_read)
    data = np.zeros((num_blocks_to_read, 784, 1), dtype=float)

    if image_blocks:
        for i, block in enumerate(image_blocks):
            for j in range(len(block)):
                if block[j] == ' ':
                      data[i][j][0] = 0 
                elif block[j] == '+':
                     data[i][j][0] = 0.5
                else:
                    data[i][j][0] = 1

            print(f"\nBlock {i + 1} (shape: {data[i].shape}):")
            print(data[i])
            # You can now work with each 'block' which is a 784x1 NumPy matrix of characters
    else:
        print("No image blocks were read.")
"""



