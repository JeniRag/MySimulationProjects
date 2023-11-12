# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:28:37 2023

@author: Sujeni
"""
import numpy as np
from Objects import *
import matplotlib.pyplot as plt

class Pixels():
    def __init__(self,h ,w):
        self.ar = w/h #aspect ration to scale values between 1 and -1
        self.x = np.linspace(-1,1,w)
        self.y = np.linspace(1/self.ar, -1/self.ar, h)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.width = w
        self.height = h
        
        self.iobj = np.full((self.height, self.width), np.nan) #index of object
        self.iHit = np.zeros((self.height,self.width, 3)) #coordinates where the ray hits object
        self.iN = np.zeros((self.height, self.width, 3)) #normal of objects
        
        self.matrix = np.zeros((self.X.shape[0], self.X.shape[1], 3))
        self.matrix[:,:,0] = self.X
        self.matrix[:,:,1] = self.Y
        self.matrix[:,:,2] = 0 #assume image screen on Z=0.
        return
    
def compute_primary_ray(ix, iy, O):
    #O np.array
    #primary ray: ray from origin (camera) through pixel
    
    #parallaizable
    d = np.array([ix, iy, 0] )- O
    
    return d/np.linalg.norm(d) #normalized ray

def compute_primary_ray_matrix(pixel_matrix,O):
    d = np.zeros_like(pixel_matrix)
    d[:,:,0] = pixel_matrix[:,:,0]-O[0] #x
    d[:,:,1] = pixel_matrix[:,:,1] - O[1]
    d[:,:,2] = pixel_matrix[:,:,2] - O[2]
    
    d_norm = np.linalg.norm(d, axis=2)
    return d / d_norm[:,:, np.newaxis]
    

def intersects_sphere(obj, prim_ray, O):
    
    a = np.dot(prim_ray, prim_ray) 
    b = 2*np.dot((O - obj.center), prim_ray)
    c = np.dot((O - obj.center),( O - obj.center)) - obj.radius**2
    D=b**2-4*a*c
   
    if D >=0: #
    
        q = -0.5 * (b + np.sign(b) * np.sqrt(D))
        t1 = q/a
        t2 = c/q
   
        if t1>0 and t2>0:
            t=min(t1,t2)
            
        
            p_Hit = prim_ray *t + O
            
            N = (p_Hit- obj.center) / obj.radius
            # N /= np.linalg.norm(N) #normalize
        
            return (p_Hit, N)   
     
    else:
        return False
    
def intersects_sphere_matrix(obj, prim_ray, O):
    #O_matrix = np.zeros_like(prim_ray_matrix)
    
    a = prim_ray[:,:,0] * prim_ray[:,:,0] + prim_ray[:,:,1] * prim_ray[:,:,1] + prim_ray[:,:,2] * prim_ray[:,:,2]  #np.dot(prim_ray, prim_ray) 
    #2D array
    OR_diff = O - obj.center
    #1D
    
    b = 2* (OR_diff[0] * prim_ray[:,:;0] + OR_diff[1] * prim_ray[:,:,1] + OR_diff[2]* prim_ray[:,:,2])
    #2D array
    
    #2*np.dot((O - obj.center), prim_ray)
    c_temp = (OR_diff[0] * OR_diff[0] + OR_diff[1] * OR_diff[1] + OR_diff[2] * OR_diff[2]) - obj.radius**2
    c= np.zeros_like(a)
    c[:,:] = c_temp 
    
    
    #np.dot((O - obj.center),( O - obj.center)) - obj.radius**2
    
    D=b**2-4*a*c
    
    cond = (D>=0) #elementes where condition is fulfilled
    
    q=np.zeros_like(a)
    t1 = np.zeros_like(a)
    t2= np.zeros_like(a)
    t=np.zeros_like(a)
    
    q[cond] = -0.5 * (b[cond] + np.sign(b)[cond] * np.sqrt(D[cond]))
    t1[cond] = q[cond] / a[cond]
    t2[cond] = c[cond] / q[cond]

    tcond = (t1 >0 and t2 >0 and D>=0)
    t[tcond] = np.min((t1, t2), axis=0)
    

    pHit = np.zeros_like(prim_ray)
        
    #pHit = prim_ray * t + O
    
    if D >=0: #
    
        q = -0.5 * (b + np.sign(b) * np.sqrt(D))
        t1 = q/a
        t2 = c/q
   
        if t1>0 and t2>0:
            t=min(t1,t2)
            
        
            p_Hit = prim_ray *t + O
            
            N = (p_Hit- obj.center) / obj.radius
            # N /= np.linalg.norm(N) #normalize
        
            return (p_Hit, N)   
     
    else:
        return False
    
    
    
def ray_casting(pixel_object, objects, O):

    #y: vertical, x: horizontal

    
    
    for i in range(pixel_object.height):
        for j in range(pixel_object.width):
            
            ix = pixel_object.X[i][j]
            iy = pixel_object.Y[i][j]
            prim_ray = compute_primary_ray(ix, iy, O) #primary ray
            
            min_dist = np.inf #initiate value for minimal distance
            
            for obj_idx, obj in enumerate(objects):
                intersecting = intersects_sphere(obj, prim_ray, O)
                if intersecting != False:
                    p_Hit, N = intersecting
                    dist = np.linalg.norm(O-p_Hit) # distance between camera point and hitting point
                    
                    
                    if dist < min_dist :
                        min_dist = dist #update minimal distance
                        pixel_object.iobj[i][j] = obj_idx  #save which object is visible in the pixel
                        pixel_object.iHit[i][j]= p_Hit
                        pixel_object.iN[i][j] = N
                            
    return pixel_object

def ray_casting_matrix(pixel_object, objects, O):
    #y: vertical, x: horizontal

    prim_ray_matrix = compute_primary_ray_matrix(pixel_object.matrix, O)
    
    for i in range(pixel_object.height):
        for j in range(pixel_object.width):

            prim_ray = prim_ray_matrix[i,j,:]
            min_dist = np.inf #initiate value for minimal distance
            
            for obj_idx, obj in enumerate(objects):
                intersecting = intersects_sphere(obj, prim_ray, O)
                if intersecting != False:
                    p_Hit, N = intersecting
                    dist = np.linalg.norm(O-p_Hit) # distance between camera point and hitting point
                    
                    
                    if dist < min_dist :
                        min_dist = dist #update minimal distance
                        pixel_object.iobj[i][j] = obj_idx  #save which object is visible in the pixel
                        pixel_object.iHit[i][j]= p_Hit
                        pixel_object.iN[i][j] = N
                            
    return pixel_object

def compute_illumination(L, pixel_object, objects):
    # L light position
    
    pixel_color = np.zeros((pixel_object.height, pixel_object.width, 3), dtype = float)

                           
    for i in range(pixel_object.height):
        for j in range(pixel_object.width):
            
            obj_idx = pixel_object.iobj[i][j]
            
            if np.isnan(obj_idx) == False :#object visible
                obj_idx = int(obj_idx)
               
                shadow_direction = L - pixel_object.iHit[i][j]
                shadow_direction /= np.linalg.norm(shadow_direction) #normalize
                shadow_origin = pixel_object.iHit[i][j] + pixel_object.iN[i][j]*1e-1
                is_shadow = False #initial assumption nothin in the way
                
                for other_obj in objects:
                    
                    intersecting = intersects_sphere(other_obj, shadow_direction, shadow_origin)
                    if intersecting !=False: #intersecting=True -> in shadow
                        is_shadow=True
                        pixel_color[i][j][:] = np.zeros(3) #black
                        break
                
                #if after iteration it's not in shadow
                if is_shadow == False:
                    # print("colored")
                    pixel_color[i][j][:] = objects[obj_idx].color
                   
                
    return pixel_color

def is_point_in_sphere(P, S, R):
    #point in sphere if R >= distance
    dist = np.linalg.norm(P-S)
    return R>= dist
#%%

#screen space at z=0
#camerate at z<=0
#light at z>= 0

# sphere1 = Sphere(np.array([10,10, 20]), 7, np.array([250,40,10]))
# sphere2= Sphere(np.array([10,10, 7]), 3, np.array([0, 0, 250]))

# objects=[sphere1, sphere2]
# Light_source = np.array([10,30, 10])
# O = np.array([10, 10, -20])
