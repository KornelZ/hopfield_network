import csv
import cv2


path = 'images'

with open('cats.csv', 'w') as file:
    writer = csv.writer(file)
    for x in range(1, 4):
        print('{}/cat{}.jpg'.format(path, x))
        image = cv2.imread('{}/cat{}.jpg'.format(path, x), 0)
        image = image.flatten()
        image = [1 if (x / 255.0) > 0.5 else -1 for x in image]
        writer.writerow(image)
