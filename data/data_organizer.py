import idx2numpy
import numpy as np
import os
import cv2  

training_images_file = 'data/train-images.idx3-ubyte'
training_images_file = idx2numpy.convert_from_file(training_images_file)

training_label_file = 'data/train-labels.idx1-ubyte'
training_label_file = idx2numpy.convert_from_file(training_label_file)

test_images_file = 'data/t10k-images.idx3-ubyte'
test_images_file = idx2numpy.convert_from_file(test_images_file)

test_label_file = 'data/t10k-labels.idx1-ubyte'
test_label_file = idx2numpy.convert_from_file(test_label_file)

# print(label_file[10])
# cv2.imshow("Image", arr_file[10])  
# cv2.waitKey(0)  # wait for a key press
# cv2.destroyAllWindows()  # close the window

training_images = 'data/training_images.txt'
training_labels = 'data/training_labels.txt'
test_images = 'data/test_images.txt'
test_labels = 'data/test_labels.txt'

if not os.path.exists(training_images): 
    np.savetxt(training_images, training_images_file.reshape(training_images_file.shape[0], -1), fmt='%d')
else:
    print(f"File {training_images} already exists.")

if not os.path.exists(training_labels): 
    np.savetxt(training_labels, training_label_file.reshape(training_label_file.shape[0], -1), fmt='%d')
else:
    print(f"File {training_labels} already exists.")

if not os.path.exists(test_images): 
    np.savetxt(test_images, test_images_file.reshape(test_images_file.shape[0], -1), fmt='%d')
else:
    print(f"File {test_images} already exists.")

if not os.path.exists(test_labels): 
    np.savetxt(test_labels, test_label_file.reshape(test_label_file.shape[0], -1), fmt='%d')
else:
    print(f"File {test_labels} already exists.")
