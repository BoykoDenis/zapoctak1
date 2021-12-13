class Box:
    def __init__(self, positionX, positionY, PositionZ, width, lenght, hight) -> None:
        self.x = positionX
        self.y = positionY
        self.z = PositionZ
        self.width = width
        self.lenght = lenght
        self.hight = hight
        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0
        self.animation = False


    def distance(self, pointX, pointY, pointZ):
        return max(abs(self.x - pointX) - self.width//2,
                   abs(self.y - pointY) - self.lenght//2,
                   abs(self.z - pointZ) - self.hight//2)
