class Sphere:
    def __init__(self, positionX, PositionY, PositionZ, radius) -> None:
        self.x = positionX
        self.y = PositionY
        self.z = PositionZ
        self.radius = radius
        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0
        self.animation = False

    def distance(self, pointX, pointY, pointZ):
        return (((pointX - self.x)**2 + (pointY - self.y)**2 + (pointZ + self.z)**2)**0.5) - self.radius

    def normal(self):
        pass
