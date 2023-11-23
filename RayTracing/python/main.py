# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:50:38 2023

@author: Sujeni
"""

import numpy as np
from Objects import *
from ray_tracing import *
import matplotlib.pyplot as plt

import time

if __name__=="__main__":
    # -z -> towards us
    # -y -> up
    # -x -> left
 
    obj1=Sphere(np.array([-0.7,0.0, 0.8]), 0.1, np.array([0,1,0])) #green 
    obj2=Sphere(np.array([-0.1, 0.0, 0.8]), 0.2, np.array([1,0,0])) #red
    obj3=Sphere(np.array([0.5, 0.0, 0.8]), 0.4, np.array([0,0,1])) #blue
    objects=[obj1, obj2, obj3]


    O=np.array([0.0, 0.0, -1.0]) #origin (camera position)
    Light_source=np.array([-1.0,0.0, 0.8])
    # Light_source=np.array([-1.0,0.7, 0.8])
    
    #make sure light source is not inside the object
    for obj in objects:
        assert is_point_in_sphere(Light_source, obj.center, obj.radius)==False

    # x= np.arange(0,20, 0.1)
    # y= np.arange(0,20,0.1)

    h=400
    w=400

    #with loop
    pixel_object_loop = Pixels(h,w)
    
    start = time.time()
    pixel_traced_loop = ray_casting(pixel_object_loop, objects, O)
    pixel_color=compute_illumination(Light_source,pixel_traced_loop, objects)
    end = time.time()
    print(f"with loops: {np.round(end-start,2)} s")

    plt.figure()
    plt.title("Loop")
    plt.imshow(pixel_color)
    
    
    #with matrixes
    pixel_object = Pixels(h,w)
    start =  time.time()
    pixel_traced = ray_casting_matrix(pixel_object, objects, O)
    
   
    pixel_color = compute_illumination_matrix(Light_source, pixel_traced, objects)
    end = time.time()
    print(f"with matrix: {np.round(end-start,2)} s")
    plt.figure()
    plt.title("Matrix")
    plt.imshow(pixel_color)
    
    # intermediate_color = np.zeros_like(pixel_traced.iHit)
    # for i in range(intermediate_color.shape[0]):
    #     for j in range(intermediate_color.shape[1]):
    #         idx = pixel_traced.iobj[i][j] 
    #         if np.isnan(idx) == False:
    #             idx = int(idx)
    #             intermediate_color[i][j]=objects[idx].color
    # plt.figure()        
    # plt.imshow(intermediate_color)
    
    
    