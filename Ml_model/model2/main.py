import cv2
import pickle
import numpy as np
import requests
import time
import asyncio
import aiohttp

url = 'http://localhost:8000/model/slotcount'


# Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 1  # Adjust thickness as needed
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 1  # Adjust thickness as needed

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cv2.putText(img, str(count), (x, y + height - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=thickness)

    cv2.putText(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 0), thickness=5)

    return spaceCounter

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    data = checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    print("data:",{"empty slots " : data})

    try:
        response = requests.post(url, json={"empty_slots":data})

        if response.status_code == 200:
            print('POST request successful!')
        print("response " , response.text)
        time.sleep(0.5)
    
    # Handle exceptions
    except Exception as e:
        print(f'An error occurred: {e}')

    # time.sleep(1)
    cv2.waitKey(10)
