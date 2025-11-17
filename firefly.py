import numpy as np
import cv2

def fitness(gray, t):
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    return (binary > 0).sum()

def firefly_optimize_threshold(gray, n=10, gen=10):
    fireflies = np.random.randint(20, 235, n)
    light = np.array([fitness(gray, t) for t in fireflies])

    for g in range(gen):
        for i in range(n):
            for j in range(n):
                if light[j] > light[i]:
                    fireflies[i] = (fireflies[i] + fireflies[j]) // 2
                    light[i] = fitness(gray, fireflies[i])

    return int(fireflies[np.argmax(light)])
