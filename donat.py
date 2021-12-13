class Donat:
    def __init__(self, R, r) -> None:
        self.R = R # big torus radius
        self.r = r # small torus radius
        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0
        self.animation = False


    def distance(self, pointX, pointY, pointZ):
        xs = pointX**2
        ys = pointY**2
        zs = pointZ**2
        rs = self.R**2
        return ( ( xs + ys + zs + rs - 2 * self.R * ((xs + ys)**(1/2)) )**(1/2) ) - self.r

    def normal_vector(self, x,y,z):
        pass



#rotation distance implementation -> seve the rotation matrix of the object
#and then apply it on the point. After just calculate the distance as usual

#NOTE: DO NOT ROTATE OBJECTS, SAVE ROTATION VALUES AND APPLY NEGATIVE ROTATION MATRIX ON THE POINT