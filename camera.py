import os
import math

class Camera:
    def __init__(self):
        self.camerax = 0
        self.cameray = 0
        self.cameraz = 0
        self.camerarotX = 0
        self.camerarotY = 0
        self.camerarotZ = 0
        self.scene = []
        self.raylimit = 100
        self.resolution = [102, 70]
        self.centerX = self.resolution[0] // 2
        self.centerY = self.resolution[1] // 2
        self.eyedistance = 10
        self.frame = [[' ' for _ in range(self.resolution[0])] for _ in range(self.resolution[1])]

    def ray_marching(self):

        for y in range(self.resolution[1]):
            for x in range(self.resolution[0]):
                x = self.centerX - x
                y = self.centerY - y
                z = self.eyedistance
                distance2point = (x**2 + y**2 + z**2)**0.5 # TODO convert from relative to global system of coordinates

                for i in range(self.raylimit):
                    self.__min_distance(x, y, z)


    def __min_distance(self, x, y, z):
        minimal = float('inf')
        for object in self.scene:
            distance = object.distance(x, y, z)
            if distance < minimal:
                minimal = distance
        return minimal #TODO implement normal vector returnal




    def next_frame(self):
        os.system('cls')
        for object in self.scene:
            if object.animation:
                object.animation_step()