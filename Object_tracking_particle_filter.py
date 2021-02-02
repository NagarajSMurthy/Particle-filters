# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:29:37 2020

@author: nagar
"""

import numpy as np
import random
from math import *
import cv2

cam = cv2.VideoCapture(0)

frame_size = (480,640,3)

world_x = 640
world_y = 480

lower = (0,125,233)
upper = (255,255,255)

(minX, minY) = (np.inf, np.inf)
(maxX, maxY) = (-np.inf, -np.inf)

max_ = [(10,10),(470,630)]
max_distance = np.sqrt((max_[1][0]-max_[0][0])**2 + (max_[1][1]-max_[0][1])**2)

landmarks = []
for i in range(1,10):    
    landmarks.append((int(frame_size[0]/10),int(i*(frame_size[1]/10))))  

for j in range(1,10):
    landmarks.append((int(9*(frame_size[0]/10)),int(j*(frame_size[1]/10))))
    
for k in range(1,10):
    landmarks.append((int(k*(frame_size[0]/10)),int((frame_size[1]/10))))
    
for l in range(1,10):
    landmarks.append((int(l*(frame_size[0]/10)),int(9*(frame_size[1]/10))))

#print('landmarks:',landmarks)
white_board = np.ones(shape = frame_size, dtype=np.uint8)

# measurement noise used in sense function
measurement_noise = 0.1
#Standard deviation to spread the particles in the prediction phase.
std = 5

class ParticleFilter:
    
    def __init__(self, width, height):
        self.x = random.random() * width # initial x position
        self.y = random.random() * height # initial y position
        self.measurement_noise = 0.0
        
    def set(self, new_x, new_y):
        self.x = float(new_x)
        self.y = float(new_y)
        
    def set_noise(self, new_d_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.distance_noise = float(new_d_noise)
        
    def measurement_prob(self, measurements):

        # calculate the correct measurement
        predicted_measurements = self.sense() 


        # compute errors
        error = 1.0
        for i in range(len(measurements)):
            error_measurement = abs(measurements[i] - predicted_measurements[i])
            error_measurement = max_distance - error_measurement 
            

            # update Gaussian
            #error *= (exp(- (error_measurement ** 2) / (self.measurement_noise ** 2) / 2.0) /  
            #          sqrt(2.0 * pi * (self.measurement_noise ** 2)))
            error *= error_measurement
            error += 1.e-300
            
        return error
    
    def move(self,velocity,deviation):
        
        
        x_ = velocity[0] + random.gauss(self.x,deviation)  #predict the X coord
        y_ = velocity[1] + random.gauss(self.y,deviation)  #predict the Y coord
        
        result = ParticleFilter(world_x,world_y)
        result.set(x_,y_)
        
        return result
    
    def sense(self):
        
        Z = []
        for i in range(len(landmarks)):
            Z.append(np.sqrt((landmarks[i][0]-self.x)**2 + (landmarks[i][1]-self.y)**2))
            
        return Z
    
def calc_distance(a,b):
    x1,y1 = a
    x2,y2 = b
    
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def get_position(p):
    x = 0.0
    y = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
    return [x / len(p), y / len(p)]

def particle_filter(velocity, measurements, white_board, N=500): 
    
    # create particles
    p = []
    for i in range(N):
        r = ParticleFilter(world_x, world_y)
        p.append(r)
    
    '''for j in range(N):
        x_,y_ = p[j].x, p[j].y
        cv2.circle(white_board, (int(x_), int(y_)), 2,(0, 0, 255), 2)'''
    
    #cv2.imshow('Random particles',white_board)
    # Update particles
    # motion update (prediction)
    p2 = []
    for i in range(N):
        p2.append(p[i].move(velocity,std))
    p = p2

    # measurement update
    w = []
    for i in range(N):
        w.append(p[i].measurement_prob(measurements))

    # resampling
    p3 = []
    
    w_sum = np.sum(w)
    w = w/w_sum
    p3 = np.random.choice(p,N,True,w)
    '''index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p3.append(p[index])'''
    p = p3
    
    
    return p, w, get_position(p)

while True:
    ret, frame = cam.read()
   
    if ret is False:
        continue
    
    image = frame.copy()
    
    blurred = cv2.GaussianBlur(image,(11,11),0)
    #cv2.imshow('Gaussian',blurred)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    #cv2.imshow('hsv',hsv)
    # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    # Masking
    mask = cv2.inRange(hsv,lower,upper)
    #mask = cv2.erode(mask,kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    #mask = cv2.dilate(mask, kernel, iterations=4)
    #res = cv2.bitwise_and(img, img, mask=mask)
    
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
 
	# only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Both of these are same
        #print((x,y))
        #print('Center from moments:',center)
        
		# only proceed if the radius meets a minimum size
        if radius > 3:
            # draw the circle and centroid on the frame,
            
            cv2.circle(frame, (int(x), int(y)), int(radius),(255, 0, 0), 2)
            #cv2.circle(frame, center, 5, (255, 0, 0), -1)
    
        measurements = []
        for i in range(len(landmarks)):
            measurements.append(calc_distance(center,landmarks[i]))
        
        velocity = (0,0)
        white_board = np.ones(shape = frame_size, dtype=np.uint8)
        
        p, w, mean = particle_filter(velocity, measurements,white_board,2000)
        
        x_mean,y_mean = mean
        cv2.circle(image, (int(x_mean), int(y_mean)), 2,(0, 0, 0), 5)
        
        weight_index = np.argsort(w)
        for i in range(1,200):
            index = weight_index[-i]
            x_,y_ = p[index].x, p[index].y
            cv2.circle(frame, (int(x_), int(y_)), 2,(0, 255, 0), 2)
        
    for landmark in landmarks:
        cv2.circle(frame,(landmark[1],landmark[0]),3,(0,0,255),3)
    
    cv2.imshow('Tracking with particle filter',image)
    cv2.imshow('masked',mask)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()


