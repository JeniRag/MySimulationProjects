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
    """
    

    Parameters
    ----------
    ix : float.
    iy : float.
    O : np.array (shape 1x3), The camera coordinates

    Returns
    -------
    np.array
        returns the primary ray normalized to 1.

    """
    #O np.array
    #primary ray: ray from origin (camera) through pixel
    
    #parallaizable
    d = np.array([ix, iy, 0] )- O
    
    return d/np.linalg.norm(d) #normalized ray

def compute_primary_ray_matrix(pixel_matrix,O):
    """
    

    Parameters
    ----------
    pixel_matrix : TYPE
        DESCRIPTION.
    O : TYPE
        The camera coordinates.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    d = np.zeros_like(pixel_matrix)
    d[:,:,0] = pixel_matrix[:,:,0]-O[0] #x
    d[:,:,1] = pixel_matrix[:,:,1] - O[1]
    d[:,:,2] = pixel_matrix[:,:,2] - O[2]
    
    d_norm = np.linalg.norm(d, axis=2)
    return d / d_norm[:,:, np.newaxis]
    

def intersects_sphere(obj, prim_ray, O):
    """
    

    Parameters
    ----------
    obj : Sphere Class
        DESCRIPTION.
    prim_ray : np.array (shape 1x3)
        the primary rays.
    O : np.array (shape 1x3)
        camera coordinates.

    Returns
    -------
    bool or tuple
        DESCRIPTION.

    """
    
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
    
def intersects_sphere_matrix(obj, prim_ray, O_matrix):
    print(f" prim_ray: {np.any(np.isnan(prim_ray))==False}")
    print(prim_ray)
    a = prim_ray[:,:,0] * prim_ray[:,:,0] + prim_ray[:,:,1] * prim_ray[:,:,1] + prim_ray[:,:,2] * prim_ray[:,:,2]  #np.dot(prim_ray, prim_ray) 
    
    
    print(f" a: {np.any(np.isnan(a))==False}")
    
    #2D array
    OR_diff = O_matrix - obj.center
    #1D
    
    b = 2* (OR_diff[:,:, 0] * prim_ray[:,:,0] + OR_diff[:,:,1] * prim_ray[:,:,1] + OR_diff[:,:,2]* prim_ray[:,:,2])
    #2D array
    
    #2*np.dot((O - obj.center), prim_ray)
    c = (OR_diff[:,:,0] * OR_diff[:,:,0] + OR_diff[:,:,1] * OR_diff[:,:,1] + OR_diff[:,:,2] * OR_diff[:,:,2]) - obj.radius**2
   
    
    D=b**2-4*a*c
    # print(D)
    
    cond = (D>=0) #elementes where condition is fulfilled
    # print(f"any D >=0: {np.any(D>=0)}")
    
    q=np.full_like(a, np.nan)
    t1 =np.full_like(a, np.nan)
    t2= np.full_like(a, np.nan)
    t=np.full_like(a, np.nan)
    
    q[cond] = -0.5 * (b[cond] + np.sign(b)[cond] * np.sqrt(D[cond]))
    t1[cond] = q[cond] / a[cond]
    t2[cond] = c[cond] / q[cond]

    tcond = ((t1 >0) & (t2 >0) & (D>=0))
    t[tcond] = np.min((t1[tcond], t2[tcond]), axis=0)
    t_matrix = np.full_like(prim_ray, np.nan)
    t_matrix[:,:,0] = t
    t_matrix[:,:,1] = t
    t_matrix[:,:,2] = t
    
    pHit = np.full_like(prim_ray, np.nan)
    pHit[tcond, :] = prim_ray[tcond,:] + t_matrix[tcond,:] + O_matrix[tcond,:]
    
    N = np.full_like(prim_ray, np.nan)
    N[tcond, :] =( pHit[tcond, :] - obj.center) / obj.radius
    #normalize?
   
    print(f"t cond {np.any(tcond == True)}")
    return pHit, N, tcond
     
    
    
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
    min_dist = np.full_like(prim_ray_matrix[:,:,0], np.inf)
    
    O_matrix = np.full_like(prim_ray_matrix, np.nan)
    O_matrix[:,:,0] = O[0]
    O_matrix[:,:,1] = O[1]
    O_matrix[:,:,2] = O[2]
    
    for obj_idx, obj in enumerate(objects):
        pHit, N, is_intersecting = intersects_sphere_matrix(obj, prim_ray_matrix, O_matrix)
        print(f"ray casting : {np.any(is_intersecting == True)}")
        
        # not_intersecting = np.isnan(pHit[:,:,0]) #condition same as _
        # is_intersecting = ~not_intersecting
        
        dist = np.full_like(min_dist, np.inf)
        
        dist[is_intersecting] = np.linalg.norm((O_matrix - pHit), axis=2)[is_intersecting]
        
        is_smaller = dist < min_dist
        
        update = is_intersecting * is_smaller
        
        min_dist[update] = dist[update]
        pixel_object.iobj[update] = obj_idx
        pixel_object.iHit[update, :] = pHit[update]
        pixel_object.iN[update, :] = N[update]
                                 
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


def compute_illumination_matrix(L, pixel_object, objects):
    # L light position
    
    L_matrix = np.zeros_like(pixel_object.iHit)
    L_matrix[:,:,0] = L[0]
    L_matrix[:,:,1] = L[1]
    L_matrix[:,:,2] = L[2]
    
    pixel_color = np.zeros((pixel_object.height, pixel_object.width, 3), dtype = float)
    
    not_visible = np.isnan(pixel_object.iobj) #condition
    is_visible = ~not_visible
    print(f"any are visible {np.any(is_visible == True)}")
    
    shadow_direction = np.full_like(pixel_object.iHit, np.nan)
    shadow_origin = np.full_like(pixel_object.iHit, np.nan)
    
    shadow_direction[is_visible,:] = L_matrix[is_visible,:] - pixel_object.iHit[is_visible,:] 
   
    norm = np.linalg.norm(shadow_direction, axis=2)[:,:, np.newaxis]
    shadow_direction /=norm

    print(f"shadow norm: {np.any(np.isnan(norm) == False)}")
    print(f"shadow normalized: {np.any(np.isnan(shadow_direction) == False)}")
    
    shadow_origin[is_visible,:] = pixel_object.iHit[is_visible,:] + pixel_object.iN[is_visible,:] *1e-1
    
    is_shadow = np.full_like(pixel_object.iobj, False, dtype = bool)
    
    for other_obj in objects:
        _, _, is_intersecting = intersects_sphere_matrix(other_obj, shadow_direction, shadow_origin)
        # print(f"any not nans: {np.any(np.isnan(shadow_origin) == False)}")
        print(f"illumination any intersecting: {np.any(is_intersecting)==True}")
        
        is_shadow[is_intersecting] = True
        pixel_color[is_intersecting,:] = np.zeros(3)
        # print(np.any(is_shadow == True))
        #pixel color stays black
        
    not_shadow = ~is_shadow
    print(f"There are any shadows: {np.any(is_shadow == True)}")
    # find a way to avoid for loop
    cond = not_shadow * is_visible
    print(f"There are visible and not visible: {np.any(cond == False) * np.any(cond == True)}")
    for i in range(pixel_object.height):
        for j in range(pixel_object.width):
            if cond[i][j]:
                obj_idx = int(pixel_object.iobj[i][j])
                pixel_color[i,j,:] = objects[obj_idx].color
           
                
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
