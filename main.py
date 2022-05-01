import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def outputShear(x, bearingLoad, gearLoad):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return bearingLoad
    elif 2.75 <= x < 4.625:
        return bearingLoad - gearLoad
    else:
        return 0.0


def inputShear(x, bearingLoad, gearLoad):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return bearingLoad
    elif 5.25 <= x < 7.125:
        return bearingLoad - gearLoad
    else:
        return 0.0


def outputMoment(x, shearLoad, mMax):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return shearLoad * (x - 0.875)
    elif 2.75 <= x < 4.625:
        return mMax - shearLoad * (x - 2.75)
    else:
        return 0.0


def inputMoment(x, shearLoad, mMax):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return shearLoad * (x - 3.375)
    elif 5.25 <= x < 7.125:
        return mMax - shearLoad * (x - 5.25)
    else:
        return 0.0


def outputTorqueDiagram(x, torque):
    if x < 2.75:
        return 0.0
    elif 2.75 <= x < 7:
        return torque
    else:
        return 0.0


def inputTorqueDiagram(x, torque):
    if x < 1:
        return 0.0
    elif 1 <= x < 5.25:
        return torque
    else:
        return 0.0


def standardDiameter(d):
    standardDiameters = [0.375, 0.5, 0.625, 0.6875, 0.75, 0.859375, 0.875, 0.984375, 1, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5]
    i = 0
    if d < standardDiameters[0]:
        return standardDiameters[0]
    while d > standardDiameters[i]:
        if d < standardDiameters[i + 1]:
            return standardDiameters[i + 1]
        i += 1


def readFigures(dRatio, rRatio):
    fig2 = mpimg.imread("A-15-8.png")
    fig1 = mpimg.imread("A-15-9.png")
    fig1plot = plt.imshow(fig1)
    plt.show()
    str1 = f"Enter Kt value from Fig.1 using D/d = {dRatio} and r/d = {rRatio}: "
    Kt = float(input(str1))
    fig2plot = plt.imshow(fig2)
    plt.show()
    str2 = f"Enter Kts value from Fig.2 using D/d = {dRatio} and r/d = {rRatio}: "
    Kts = float(input(str2))
    return Kt, Kts


# Design variables
outputPower = 150 #HP
inputSpeed = 6700 #rpm
outputSpeed = 30000 #rpm
outputMinimum = outputSpeed - 0.01 * outputSpeed

# Material Properties
materialName = "1015 CD Steel"
tensileStrength = 56 #ksi
yieldStrength = 47 #ksi

# Shaft Properties
shaftLength = 8 #inches

# Gear Properties
helixAngle = 30 #degrees
psi = math.radians(helixAngle)
normalPressureAngle = 20 #degrees
phi_n = math.radians(normalPressureAngle)
phi_t = math.atan(math.tan(phi_n) / math.cos(psi))
transversePressureAngle = math.degrees(phi_t)
#available in 0.5 inch width increments
Ks = 1
Cf = 1

# Reliability
shaftLifetime = "infinite"
gearReliability = 0.9
gearLifetime = 1000
bearingReliability = gearReliability
bearingLifetime = gearLifetime

# Gear Box Properties
maxBoxWidth = 15 #inches
wallThickness = 1.5 #inches


# Calculate the gear teeth and rotation speed
k = 1
gearRatio = outputSpeed / inputSpeed
pinionTeeth = (2 * k * math.cos(psi) * (gearRatio + math.sqrt((gearRatio ** 2) + (1 + 2 * gearRatio) * (math.sin(phi_t) ** 2)))) / ((1 + 2 * gearRatio) * (math.sin(phi_t) ** 2))
wholePinionTeeth = round(pinionTeeth)
gearTeeth = round(gearRatio * wholePinionTeeth)
actualRatio = gearTeeth / wholePinionTeeth
actualOutputSpeed = inputSpeed * actualRatio

print("Teeth Calculations")
print("Number of Pinion Teeth:", wholePinionTeeth)
print("Number of Gear Teeth:", gearTeeth)
print("Output Speed:", actualOutputSpeed, "rpm")
print("Minimum Output Speed:", outputMinimum, "rpm")
print("")

# Calculate torque in the gears
pinionTorque = outputPower / actualOutputSpeed * 33000 / (2 * math.pi)
gearTorque = outputPower / inputSpeed * 33000 / (2 * math.pi)

print("Torque Calculations")
print("Torque on Pinion:", pinionTorque, "lbf * ft")
print("Torque on Gear:", gearTorque, "lbf * ft")
print("")

# Calculate diametral pitch and gear diameters
pitchOptions = [2, 2.25, 2.5, 3, 4, 6, 8, 10, 12, 16]
P = (wholePinionTeeth + gearTeeth + 2) / (maxBoxWidth - wallThickness)
diametralPitch = 0
maxV = 10000 #ft/min
maxPinionDiameter = 12 * maxV / (math.pi * actualOutputSpeed)
i = 0
pinionDiameter = 100
while pinionDiameter > maxPinionDiameter:
    diametralPitch = pitchOptions[i]
    pinionDiameter = wholePinionTeeth / diametralPitch
    i += 1
pinionDiameter = wholePinionTeeth / diametralPitch
gearDiameter = gearTeeth / diametralPitch
# Calculate gear box minimum width here to show that it fits in the range

print("Pitch and Diameter Calculations")
print("Diametral Pitch:", diametralPitch)
print("Pinion Diameter:", pinionDiameter)
print("Gear Diameter:", gearDiameter)
print("")

# Calculate pitch line velocity and gear loads
pitchLineVelocity = math.pi * pinionDiameter * actualOutputSpeed / 12
transverseLoad = 33000 * outputPower / pitchLineVelocity
radialLoad = transverseLoad * math.tan(phi_t)
axialLoad = transverseLoad * math.tan(psi)

print("Output Gear Loads")
print("Pitch Line Velocity, V:", pitchLineVelocity)
print("Wt:", transverseLoad, "lbf")
print("Wr:", radialLoad, "lbf")
print("Wa:", axialLoad, "lbf")
print("")

# Calculated loads on the bearings of the output shaft
RCz = 0.5 * transverseLoad
RCy = 0.5 * radialLoad
RCx = axialLoad
print("Bearing C:")
print("Transverse:", RCz, "lbf")
print("Radial:", RCy, "lbf")
print("Axial:", RCx, "lbf")
print("")

RDz = RCz
RDy = RCy
RDx = 0
print("Bearing D:")
print("Transverse:", RDz, "lbf")
print("Radial:", RDy, "lbf")
print("Axial:", RDx, "lbf")
print("")

# Create Shear, Moment, and Torque Diagrams for the output shaft
maxShear = math.sqrt(RCy ** 2 + RCz ** 2)
maxMoment = maxShear * 1.875
outputShaftTorque = pinionTorque * 12
'''
x_y = np.linspace(0, 8, 1000)
x_z = np.linspace(0, 8, 1000)
vy = np.vectorize(outputShear)
y_y = vy(x_y, RCy, radialLoad)
vz = np.vectorize(outputShear)
y_z = vz(x_z, RCz, transverseLoad)

plt.title("Output Shaft Shear Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Shear Force (lbf)")
plt.plot(x_y, y_y, label="V_y")
plt.plot(x_z, y_z, label="V_z")
plt.legend()
plt.show()
'''
'''
x_m = np.linspace(0, 8, 1000)
vm = np.vectorize(outputMoment)
y_m = vm(x_m, maxShear, maxMoment)

plt.title("Output Shaft Moment Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Bending Moment (lbf*in)")
plt.plot(x_m, y_m, label="Moment")
plt.show()
'''
'''
print("Torque:", outputShaftTorque, "lbf * in")
x_t = np.linspace(0, 8, 1000)
vt = np.vectorize(outputTorqueDiagram)
y_t = vt(x_t, outputShaftTorque)

plt.title("Output Shaft Torque Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Torque (lbf * in)")
plt.plot(x_t, y_t, label="Torque")
plt.show()
'''
# Initialize dictionaries for each point
S_ut = 56000
S_ut_ksi = S_ut / 1000
Sy = 47000
n = 1.5

ka = 2.0 * math.pow(56, -0.217)
kb = 0.9

pointDistances = [1.25, 1.75, 2.0, 3.25, 3.5, 3.75, 4.25, 5, 5.25, 6]
points = []
roundedShoulders = [1, 2, 5, 8]
sharpShoulders = [0, 6]
keyways = [3, 9]
retainingRings = [4, 7]
for i in range(10):
    name = chr(i + 81)
    xDist = pointDistances[i]
    # Concentrations listed as Kt, Kts, Kf, Kfs, (r/d), root(a)bend, root(a)tors, q, qs
    if i in roundedShoulders:
        type = "roundShoulder"
        concentrations = [1.7, 1.5, 1.7, 1.5, 0.1, 0, 0, 0, 0]
    elif i in sharpShoulders:
        type = "sharpShoulder"
        concentrations = [2.7, 2.2, 2.7, 2.2, 0.02, 0, 0, 0, 0]
    elif i in keyways:
        type = "keyway"
        concentrations = [2.14, 3, 2.14, 3]
    elif i in retainingRings:
        type = "retainingRing"
        concentrations = [5, 3, 5, 3]
    moment = outputMoment(xDist, maxShear, maxMoment)
    torque = outputTorqueDiagram(xDist, outputShaftTorque)
    Se = ka * kb * S_ut / 2
    vonMises = [0, 0]
    safetyFactors = [0, 0]
    point = {"name": name, "x": xDist, "type": type, "moment": moment, "torque": torque, "concentrations": concentrations, "Se": Se, "VonMises": vonMises, "n": safetyFactors, "d": 0 }
    points.append(point)

def minDiameter(num):
    Kf = points[num]["concentrations"][2]
    Kfs = points[num]["concentrations"][3]
    M = points[num]["moment"]
    T = points[num]["torque"]
    Se = points[num]["Se"]
    dMin = math.pow(16 * n / math.pi * (2 * Kf * M) / Se + math.sqrt(3 * (Kfs * T) ** 2) / S_ut, 1 / 3)
    dMin = standardDiameter(dMin)
    return dMin

def stressConcentrationFactors(num, d, dRatio):
    rRatio = points[num]["concentrations"][4]
    r = d * rRatio
    kt, kts = readFigures(dRatio, rRatio)

    # Kf and Kfs
    neubergBend = 0.246 - 3.08 * (10 ** -3) * S_ut_ksi + 1.51 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    neubergTors = 0.19 - 2.51 * (10 ** - 3) * S_ut_ksi + 1.35 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    q = 1 / (1 + neubergBend / math.sqrt(r))
    qs = 1 / (1 + neubergTors / math.sqrt(r))
    Kf = 1 + q * (kt - 1)
    Kfs = 1 + qs * (kts - 1)

    # Save to Dictionary
    points[num]["concentrations"][0] = kt
    points[num]["concentrations"][1] = kts
    points[num]["concentrations"][2] = Kf
    points[num]["concentrations"][3] = Kfs

    points[num]["concentrations"][5] = neubergBend
    points[num]["concentrations"][6] = neubergTors
    points[num]["concentrations"][7] = q
    points[num]["concentrations"][8] = qs

    return Kf, Kfs

def safetyFactors(num, dMin, Kf, Kfs):
    nf, ny = 0,0
    while nf < 1.5 or ny < 1.5:
        kb = 0.879 * dMin ** -0.107
        Se = ka * kb * S_ut / 2
        sig_a = 32 * Kf * points[num]["moment"] / (math.pi * dMin ** 3)
        sig_m = math.sqrt(3 * (16 * Kfs * points[num]["torque"] / (math.pi * dMin ** 3)) ** 2)
        nf = 1 / (sig_a / Se + sig_m / S_ut)
        ny = Sy / (sig_a + sig_m)
        if nf > 1.5 and ny > 1.5:
            points[num]["n"][0] = nf
            points[num]["n"][1] = ny
        else:
            dMin = standardDiameter(dMin + 0.0001)
    points[num]["Se"] = Se
    points[num]["VonMises"] = [sig_a, sig_m]
    points[num]["d"] = dMin
    return dMin


outputShaftDiameters = {"Do1": 0, "Do2": 0, "Do3": 0, "Do4": 0, "Do5": 0, "Do6": 0, "Do7": 0}

# Find minimum diameter Do4 by checking points U, T, S
dMinU = minDiameter(4)
dMinU = safetyFactors(4, dMinU, points[4]["concentrations"][2], points[4]["concentrations"][3])

dMinT = minDiameter(3)
dMinT = safetyFactors(3, dMinT, points[3]["concentrations"][2], points[3]["concentrations"][3])

dMinS = minDiameter(2)
KfS, KfsS = stressConcentrationFactors(2, dMinS, 1.1)
dMinS = safetyFactors(2, dMinS, points[2]["concentrations"][2], points[2]["concentrations"][3])

Do4 = max(dMinU, dMinT, dMinT)
outputShaftDiameters["Do4"] = Do4
Do3 = Do4 * 1.1
outputShaftDiameters["Do3"] = Do3

dMinU = safetyFactors(4, Do4, points[4]["concentrations"][2], points[4]["concentrations"][3])
dMinT = safetyFactors(3, Do4, points[3]["concentrations"][2], points[3]["concentrations"][3])
KfS, KfsS = stressConcentrationFactors(2, Do4, 1.1)
dMinS = safetyFactors(2, Do4, points[2]["concentrations"][2], points[2]["concentrations"][3])

print("")

# Find minimum diameter for Do6 by checking U and X
dMinX = minDiameter(7)
dMinX = safetyFactors(7, dMinX, points[7]["concentrations"][2], points[7]["concentrations"][3])

dMinW = minDiameter(6)
if dMinW < dMinX:
    dMinW = dMinX
equalRatio = math.sqrt(Do4 / dMinW)
KfW, KfsW = stressConcentrationFactors(6, dMinW, equalRatio)
dMinW = safetyFactors(6, dMinW, KfW, KfsW)
Do6 = dMinW
outputShaftDiameters["Do6"] = Do6

# Check safety factor at V
dMinV = equalRatio * dMinW
KfD, KfsD = stressConcentrationFactors(5, dMinV, equalRatio)
dMinV = safetyFactors(5, dMinV, KfD, KfsD)
Do5 = dMinV
outputShaftDiameters["Do5"] = Do5

# Assume that Do1 = Do6 so that the bearings are the same size then check safety factors at Q and R
leftRatio = math.sqrt(Do3 / Do6)
KfQ, KfsQ = stressConcentrationFactors(0, Do6, leftRatio)
dMinQ = safetyFactors(0, Do6, KfQ, KfsQ)
Do1 = dMinQ
outputShaftDiameters["Do1"] = Do1

dMinR = leftRatio * dMinQ
KfR, KfsR = stressConcentrationFactors(1, dMinR, leftRatio)
dMinR = safetyFactors(1, dMinR, KfR, KfsR)
Do2 = dMinR
outputShaftDiameters["Do2"] = Do2

# Check safety factor at y and z with Do7 = Do6 / 1.1
Do7 = Do6 / 1.1
dMinY = minDiameter(8)
dMinZ = minDiameter(9)
Do7 = max(Do7, dMinY, dMinZ)
rightRatio = Do6 / Do7
KfY, KfsY = stressConcentrationFactors(8, Do7, rightRatio)
dMinY = safetyFactors(8, Do7, KfY, KfsY)
dMinZ = safetyFactors(9, Do7, points[9]["concentrations"][2], points[9]["concentrations"][3])
if dMinY < dMinZ:
    safetyFactors(8, dMinZ, KfY, KfsY)
outputShaftDiameters["Do7"] = Do7


print("Points on the output shaft")
for p in points:
    print("Point", p["name"], ": Diameter:", p["d"], "Safety Factors (fatigue, yield): ", p["n"], "Von Mises Stresses (Alternating, Midrange):", p["VonMises"])

print(outputShaftDiameters)
print("")