import numpy as np

MAX = 10
X, Y, Z = -2, -10, -4
P = (X, Y, Z)

def getRGB(MAX, P):
    return tuple(MAX + i if i < 0 else i for i in P)

def getW(MAX, P):
    SPREAD = 4 * MAX
    total = sum(abs(i) for i in P)
    return SPREAD - total

def normalizeQuaternion(q):
    norm = np.linalg.norm(q)
    return tuple(i / norm for i in q)

def rgbFloat(rgb, MAX):
    return tuple(i / MAX for i in rgb)

def rgb255(rgbFloat):
    return tuple(int(round(np.clip(f * 255, 0, 255))) for f in rgbFloat)

def rgbHex(rgb255):
    return "#{:02X}{:02X}{:02X}".format(*rgb255)

C = getRGB(MAX, P)
W = getW(MAX, P)
Q = (W, X, Y, Z)
UnitQ = normalizeQuaternion(Q)

Cfloat = rgbFloat(C, MAX)
C255 = rgb255(Cfloat)
Chex = rgbHex(C255)

print("Input P:", P)
print("RGB Float:", Cfloat)
print("RGB 0–255:", C255)
print("RGB HEX:", Chex)
print("W (magnitude offset):", W)
print("Quaternion:", Q)
print("Unit Quaternion:", UnitQ)



