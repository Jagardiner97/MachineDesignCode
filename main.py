import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
    if dRatio == 1.1 and rRatio == 0.1:
        return 1.6, 1.2
    elif dRatio == 1.3142574813455419 and rRatio == 0.02:
        return 2.8, 2.2
    elif dRatio == 1.3142574813455419 and rRatio == 0.1:
        return 1.6, 1.4
    elif dRatio == 1.3784048752090223 and rRatio == 0.02:
        return 2.8, 2.2
    elif dRatio == 1.3784048752090223 and rRatio == 0.1:
        return 1.65, 1.4
    elif dRatio == 1.137592917989042 and rRatio == 0.02:
        return 2.6, 1.6
    elif dRatio == 1.193117518002609 and rRatio == 0.02:
        return 2.7, 2.2
    elif dRatio == 1.193117518002609 and rRatio == 0.1:
        return 1.65, 1.35
    elif dRatio == 1.2060453783110545 and rRatio == 0.02:
        return 2.6, 2.0
    elif dRatio == 1.2060453783110545 and rRatio == 0.1:
        return 1.65, 1.35
    elif dRatio == 1.2649110640673518 and rRatio == 0.02:
        return 2.7, 2.2
    elif dRatio == 1.2649110640673518 and rRatio == 0.1:
        return 1.65, 1.4
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
outputPower = 150  # HP
inputSpeed = 6700  # rpm
outputSpeed = 30000  # rpm
outputMinimum = outputSpeed - 0.01 * outputSpeed

# Material Properties
materialName = "1015 CD Steel"
tensileStrength = 56  # ksi
yieldStrength = 47  # ksi

# Shaft Properties
shaftLength = 8  # inches

# Gear Properties
helixAngle = 30  # degrees
psi = math.radians(helixAngle)
normalPressureAngle = 20  # degrees
phi_n = math.radians(normalPressureAngle)
phi_t = math.atan(math.tan(phi_n) / math.cos(psi))
transversePressureAngle = math.degrees(phi_t)
# available in 0.5 inch width increments
Ks = 1
Cf = 1

# Reliability
shaftLifetime = "infinite"
gearReliability = 0.9
gearLifetime = 1000
bearingReliability = gearReliability
bearingLifetime = gearLifetime

# Gear Box Properties
maxBoxWidth = 15  #inches
wallThickness = 1.5  #inches

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
maxV = 10000  # ft/min
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

# Calculate loads on the bearings of the input shaft
RAz = 1/2 * transverseLoad
RAy = (-radialLoad * 1.875 + axialLoad * gearDiameter / 2) / (1.875 * 2)
RAx = 0
print("Bearing A:")
print("Transverse: ", RAz, "lbf")
print("Radial:", RAy, "lbf")
print("Axial:", RAx, "lbf")
print("")

RBz = -RAz
RBy = -radialLoad - RAy
RBx = axialLoad
print("Bearing B:")
print("Transverse: ", RBz, "lbf")
print("Radial:", RBy, "lbf")
print("Axial:", RBx, "lbf")
print("")

# Calculated loads on the bearings of the output shaft
RCz = 0.5 * transverseLoad
RCy = (-axialLoad * pinionDiameter / 2 - radialLoad * 1.625) / (2 * 1.625)
RCx = axialLoad
print("Bearing C:")
print("Transverse:", RCz, "lbf")
print("Radial:", RCy, "lbf")
print("Axial:", RCx, "lbf")
print("")

RDz = -RCz
RDy = -radialLoad - RCy
RDx = 0
print("Bearing D:")
print("Transverse:", RDz, "lbf")
print("Radial:", RDy, "lbf")
print("Axial:", RDx, "lbf")
print("")

def outputShear(x):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return math.sqrt(RCy ** 2 + RCz ** 2)
    elif 2.75 <= x < 4.625:
        return math.sqrt(transverseLoad ** 2 + radialLoad ** 2) - math.sqrt(RCy ** 2 + RCz ** 2)
    else:
        return 0.0

def outputShearY(x):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return RCy
    elif 2.75 <= x < 4.625:
        return radialLoad - RCy
    else:
        return 0.0

def outputShearZ(x):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return RCz
    elif 2.75 <= x < 4.625:
        return RCz - transverseLoad
    else:
        return 0.0

def inputShear(x):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return math.sqrt(RAy ** 2 + RAz ** 2)
    elif 5.25 <= x < 7.125:
        return math.sqrt(transverseLoad ** 2 + radialLoad ** 2) - math.sqrt(RAy ** 2 + RAz ** 2)
    else:
        return 0.0

def inputShearY(x):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return RAy
    elif 5.25 <= x < 7.125:
        return RAy - radialLoad
    else:
        return 0.0

def inputShearZ(x):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return RAz
    elif 5.25 <= x < 7.125:
        return RAz - transverseLoad
    else:
        return 0.0


mMaxOut = outputShear(2.0) * (2.75 - 0.875)
mMaxIn = inputShear(5.25) * (5.25 - 3.375)
outputShaftTorque = pinionTorque * 12
inputShaftTorque = gearTorque * 12

def outputMoment(x):
    if x < 0.875:
        return 0.0
    elif 0.875 <= x < 2.75:
        return outputShear(x) * (x - 0.875)
    elif 2.75 <= x <= 4.625:
        return mMaxOut - outputShear(x) * (x - 2.75)
    else:
        return outputMoment(4.624)

def inputMoment(x):
    if x < 3.375:
        return 0.0
    elif 3.375 <= x < 5.25:
        return inputShear(x) * (x - 3.375)
    elif 5.25 <= x <= 7.125:
        return mMaxIn - inputShear(x) * (x - 5.25)
    else:
        return inputMoment(7.124)

def outputTorqueDiagram(x):
    if x < 2.75:
        return 0.0
    elif 2.75 <= x < 7:
        return outputShaftTorque
    else:
        return 0.0

def inputTorqueDiagram(x):
    if x < 1:
        return 0.0
    elif 1 <= x < 5.25:
        return inputShaftTorque
    else:
        return 0.0


# Create Shear, Moment, and Torque Diagrams for the output shaft
'''outputMaxShear = math.sqrt(RDy ** 2 + RDz ** 2)
outputMaxMoment = outputMaxShear * 1.875
inputMaxShear = outputMaxShear
inputMaxMoment = outputMaxMoment'''

print("")

x_y = np.linspace(0, 8, 1000)
x_z = np.linspace(0, 8, 1000)
vy = np.vectorize(outputShearY)
y_y = vy(x_y)
vz = np.vectorize(outputShearZ)
y_z = vz(x_z)

plt.title("Output Shaft Shear Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Shear Force (lbf)")
plt.plot(x_y, y_y, label="V_y")
plt.plot(x_z, y_z, label="V_z")
plt.legend()
plt.show()

x_m = np.linspace(0, 8, 1000)
vm = np.vectorize(outputMoment)
y_m = vm(x_m)

plt.title("Output Shaft Moment Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Bending Moment (lbf*in)")
plt.plot(x_m, y_m, label="Moment")
plt.show()

print("Torque:", outputShaftTorque, "lbf * in")
x_t = np.linspace(0, 8, 1000)
vt = np.vectorize(outputTorqueDiagram)
y_t = vt(x_t)

plt.title("Output Shaft Torque Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Torque (lbf * in)")
plt.plot(x_t, y_t, label="Torque")
plt.show()

# Create Shear, Moment, and Torque Diagrams for the input shaft
x_yi = np.linspace(0, 8, 1000)
x_zi = np.linspace(0, 8, 1000)
vyi = np.vectorize(inputShearY)
y_yi = vyi(x_yi)
vzi = np.vectorize(inputShearZ)
y_zi = vzi(x_zi)

plt.title("Input Shaft Shear Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Shear Force (lbf)")
plt.plot(x_yi, y_yi, label="V_y")
plt.plot(x_zi, y_zi, label="V_z")
plt.legend()
plt.show()

x_mi = np.linspace(0, 8, 1000)
vmi = np.vectorize(inputMoment)
y_mi = vmi(x_mi)

plt.title("Input Shaft Moment Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Bending Moment (lbf*in)")
plt.plot(x_mi, y_mi, label="Moment")
plt.show()

print("Torque:", outputShaftTorque, "lbf * in")
x_ti = np.linspace(0, 8, 1000)
vti = np.vectorize(inputTorqueDiagram)
y_ti = vti(x_ti)

plt.title("Input Shaft Torque Diagram")
plt.xlabel("x (in.)")
plt.ylabel("Torque (lbf * in)")
plt.plot(x_ti, y_ti, label="Torque")
plt.show()


# Initialize dictionaries for each point on the output shaft
S_ut = 56000
S_ut_ksi = S_ut / 1000
Sy = 47000
n = 1.5

ka = 2.0 * math.pow(56, -0.217)
kb = 0.9

pointDistances = [1.25, 1.75, 2.0, 3.25, 3.5, 3.75, 4.25, 5, 5.25, 6]
outputPoints = []
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
    moment = outputMoment(xDist)
    torque = outputTorqueDiagram(xDist)
    Se = ka * kb * S_ut / 2
    vonMises = [0, 0]
    safetyFactors = [0, 0]
    point = {"name": name, "x": xDist, "type": type, "moment": moment, "torque": torque, "concentrations": concentrations, "Se": Se, "VonMises": vonMises, "n": safetyFactors, "d": 0 }
    outputPoints.append(point)

def minDiameter(num):
    Kf = outputPoints[num]["concentrations"][2]
    Kfs = outputPoints[num]["concentrations"][3]
    M = outputPoints[num]["moment"]
    T = outputPoints[num]["torque"]
    Se = outputPoints[num]["Se"]
    dMin = math.pow(16 * n / math.pi * (2 * Kf * M) / Se + math.sqrt(3 * (Kfs * T) ** 2) / S_ut, 1 / 3)
    dMin = standardDiameter(dMin)
    return dMin

def stressConcentrationFactors(num, d, dRatio):
    rRatio = outputPoints[num]["concentrations"][4]
    r = d * rRatio
    if dRatio == 1:
        kt, kts = 1, 1
    else:
        kt, kts = readFigures(dRatio, rRatio)

    # Kf and Kfs
    neubergBend = 0.246 - 3.08 * (10 ** -3) * S_ut_ksi + 1.51 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    neubergTors = 0.19 - 2.51 * (10 ** - 3) * S_ut_ksi + 1.35 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    q = 1 / (1 + neubergBend / math.sqrt(r))
    qs = 1 / (1 + neubergTors / math.sqrt(r))
    Kf = 1 + q * (kt - 1)
    Kfs = 1 + qs * (kts - 1)

    # Save to Dictionary
    outputPoints[num]["concentrations"][0] = kt
    outputPoints[num]["concentrations"][1] = kts
    outputPoints[num]["concentrations"][2] = Kf
    outputPoints[num]["concentrations"][3] = Kfs

    outputPoints[num]["concentrations"][5] = neubergBend
    outputPoints[num]["concentrations"][6] = neubergTors
    outputPoints[num]["concentrations"][7] = q
    outputPoints[num]["concentrations"][8] = qs

    return Kf, Kfs

def safetyFactors(num, dMin, Kf, Kfs):
    nf, ny = 0,0
    while nf < 1.5 or ny < 1.5:
        kb = 0.879 * dMin ** -0.107
        Se = ka * kb * S_ut / 2
        sig_a = 32 * Kf * outputPoints[num]["moment"] / (math.pi * dMin ** 3)
        sig_m = math.sqrt(3 * (16 * Kfs * outputPoints[num]["torque"] / (math.pi * dMin ** 3)) ** 2)
        nf = 1 / (sig_a / Se + sig_m / S_ut)
        ny = Sy / (sig_a + sig_m)
        if nf > 1.5 and ny > 1.5:
            outputPoints[num]["n"][0] = nf
            outputPoints[num]["n"][1] = ny
        else:
            dMin = standardDiameter(dMin + 0.0001)
    outputPoints[num]["Se"] = Se
    outputPoints[num]["VonMises"] = [sig_a, sig_m]
    outputPoints[num]["d"] = dMin
    return dMin


outputShaftDiameters = {"Do1": 0, "Do2": 0, "Do3": 0, "Do4": 0, "Do5": 0, "Do6": 0, "Do7": 0}

# Find minimum diameter Do4 by checking points U, T, S
dMinU = minDiameter(4)
dMinU = safetyFactors(4, dMinU, outputPoints[4]["concentrations"][2], outputPoints[4]["concentrations"][3])

dMinT = minDiameter(3)
dMinT = safetyFactors(3, dMinT, outputPoints[3]["concentrations"][2], outputPoints[3]["concentrations"][3])

dMinS = minDiameter(2)
KfS, KfsS = stressConcentrationFactors(2, dMinS, 1.1)
dMinS = safetyFactors(2, dMinS, outputPoints[2]["concentrations"][2], outputPoints[2]["concentrations"][3])

Do4 = max(dMinU, dMinT, dMinT)
outputShaftDiameters["Do4"] = Do4
Do3 = Do4 * 1.1
outputShaftDiameters["Do3"] = Do3

dMinU = safetyFactors(4, Do4, outputPoints[4]["concentrations"][2], outputPoints[4]["concentrations"][3])
dMinT = safetyFactors(3, Do4, outputPoints[3]["concentrations"][2], outputPoints[3]["concentrations"][3])
KfS, KfsS = stressConcentrationFactors(2, Do4, 1.1)
dMinS = safetyFactors(2, Do4, KfS, KfsS)

print("")

# Find minimum diameter for Do6 by checking U and X
dMinX = minDiameter(7)
dMinX = safetyFactors(7, dMinX, outputPoints[7]["concentrations"][2], outputPoints[7]["concentrations"][3])

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
dMinZ = safetyFactors(9, Do7, outputPoints[9]["concentrations"][2], outputPoints[9]["concentrations"][3])
if dMinY < dMinZ:
    safetyFactors(8, dMinZ, KfY, KfsY)
outputShaftDiameters["Do7"] = max(Do7, dMinY, dMinZ)

# Initialize Dictionaries for each point on the input shaft "G to P"
inputShaftDiameters = {"Di1": 0, "Di2": 0, "Di3": 0, "Di4": 0, "Di5": 0, "Di6": 0, "Di7": 0}
inputPointDistances = [2.0, 2.75, 3.0, 3.75, 4.25, 4.5, 4.75, 6.0, 6.25, 6.75]
inputPoints = []
roundedShoulders = [1, 4, 7, 8]
sharpShoulders = [3, 9]
keyways = [0, 6]
retainingRings = [2, 5]
for i in range(10):
    name = chr(i + 71)
    xDist = inputPointDistances[i]
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
    moment = inputMoment(xDist)
    torque = inputTorqueDiagram(xDist)
    Se = ka * kb * S_ut / 2
    vonMises = [0, 0]
    safetyFactors = [0, 0]
    point = {"name": name, "x": xDist, "type": type, "moment": moment, "torque": torque, "concentrations": concentrations, "Se": Se, "VonMises": vonMises, "n": safetyFactors, "d": 0 }
    inputPoints.append(point)

def inputMinDiameter(num):
    Kf = inputPoints[num]["concentrations"][2]
    Kfs = inputPoints[num]["concentrations"][3]
    M = inputPoints[num]["moment"]
    T = inputPoints[num]["torque"]
    Se = inputPoints[num]["Se"]
    dMin = math.pow(16 * n / math.pi * (2 * Kf * M) / Se + math.sqrt(3 * (Kfs * T) ** 2) / S_ut, 1 / 3)
    dMin = standardDiameter(dMin)
    return dMin

def inputStressConcentrationFactors(num, d, dRatio):
    rRatio = inputPoints[num]["concentrations"][4]
    r = d * rRatio
    if dRatio == 1:
        kt, kts = 1, 1
    else:
        kt, kts = readFigures(dRatio, rRatio)

    # Kf and Kfs
    neubergBend = 0.246 - 3.08 * (10 ** -3) * S_ut_ksi + 1.51 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    neubergTors = 0.19 - 2.51 * (10 ** - 3) * S_ut_ksi + 1.35 * (10 ** -5) * (S_ut_ksi ** 2) - 2.67 * (10 ** -8) * (S_ut_ksi ** 3)
    q = 1 / (1 + neubergBend / math.sqrt(r))
    qs = 1 / (1 + neubergTors / math.sqrt(r))
    Kf = 1 + q * (kt - 1)
    Kfs = 1 + qs * (kts - 1)

    # Save to Dictionary
    inputPoints[num]["concentrations"][0] = kt
    inputPoints[num]["concentrations"][1] = kts
    inputPoints[num]["concentrations"][2] = Kf
    inputPoints[num]["concentrations"][3] = Kfs

    inputPoints[num]["concentrations"][5] = neubergBend
    inputPoints[num]["concentrations"][6] = neubergTors
    inputPoints[num]["concentrations"][7] = q
    inputPoints[num]["concentrations"][8] = qs

    return Kf, Kfs

def inputSafetyFactors(num, dMin, Kf, Kfs):
    nf, ny = 0,0
    while nf < 1.5 or ny < 1.5:
        kb = 0.879 * dMin ** -0.107
        Se = ka * kb * S_ut / 2
        sig_a = 32 * Kf * inputPoints[num]["moment"] / (math.pi * dMin ** 3)
        sig_m = math.sqrt(3 * (16 * Kfs * inputPoints[num]["torque"] / (math.pi * dMin ** 3)) ** 2)
        nf = 1 / (sig_a / Se + sig_m / S_ut)
        ny = Sy / (sig_a + sig_m)
        if nf > 1.5 and ny > 1.5:
            inputPoints[num]["n"][0] = nf
            inputPoints[num]["n"][1] = ny
        else:
            dMin = standardDiameter(dMin + 0.0001)
    inputPoints[num]["Se"] = Se
    inputPoints[num]["VonMises"] = [sig_a, sig_m]
    inputPoints[num]["d"] = dMin
    return dMin


# Find the minimum diameters for Di4 and Di5 by checking points L, M, N
dMinL = inputMinDiameter(5)
dMinL = inputSafetyFactors(5, dMinL, inputPoints[5]["concentrations"][2], inputPoints[5]["concentrations"][3])

dMinM = inputMinDiameter(6)
dMinM = inputSafetyFactors(6, dMinM, inputPoints[6]["concentrations"][2], inputPoints[6]["concentrations"][3])

dMinN = minDiameter(7)
KfN, KfsN = inputStressConcentrationFactors(7, dMinN, 1.1)
dMinN = inputSafetyFactors(7, dMinN, KfN, KfsN)

Di4 = max(dMinL, dMinM, dMinN)
inputShaftDiameters["Di4"] = Di4
Di5 = Di4 * 1.1
inputShaftDiameters["Di5"] = Di5

dMinL = inputSafetyFactors(5, Di4, inputPoints[5]["concentrations"][2], inputPoints[5]["concentrations"][3])
dMinM = inputSafetyFactors(6, Di4, inputPoints[6]["concentrations"][2], inputPoints[6]["concentrations"][3])
KfN, KfsN = inputStressConcentrationFactors(7, Di4, 1.1)
dMinN = inputSafetyFactors(7, Di4, KfN, KfsN)

# Find the minimum diameter for Di2 by checking points I and J
dMinI = inputMinDiameter(2)
dMinI = inputSafetyFactors(2, dMinI, inputPoints[2]["concentrations"][2], inputPoints[2]["concentrations"][3])

dMinJ = inputMinDiameter(3)
Di2 = max(dMinI, dMinJ)
equalIRatio = math.sqrt(Di4 / Di2)
dMinJ = Di2
KfJ, KfsJ = inputStressConcentrationFactors(3, dMinJ, equalIRatio)
dMinJ = inputSafetyFactors(3, dMinJ, KfJ, KfsJ)

dMinI = dMinJ
Di2 = dMinJ
inputShaftDiameters["Di2"] = Di2

# Check Safety factor at K
dMinK = equalIRatio * Di2
KfK, KfsK = inputStressConcentrationFactors(4, dMinK, equalRatio)
dMinK = inputSafetyFactors(4, dMinK, KfK, KfsK)
Di3 = dMinK
inputShaftDiameters["Di3"] = Di3

# Assume Di7 = Di2 so bearings are the same size and calculate diameters using points P and O
Di7 = Di2
rightIRatio = math.sqrt(Di5 / Di7)
KfP, KfsP = inputStressConcentrationFactors(9, Di7, rightIRatio)
dMinP = inputSafetyFactors(9, Di7, KfP, KfsP)
Di7 = dMinP
inputShaftDiameters["Di7"] = Di7

dMinO = rightIRatio * Di7
Di6 = dMinO
KfO, KfsO = inputStressConcentrationFactors(8, Di7, rightIRatio)
dMinO = inputSafetyFactors(8, dMinO, KfO, KfsO)
inputShaftDiameters["Di6"] = Di6

# Check safety factors and find Di1 using points G and H
Di1 = Di2 / 1.1
dMinH = inputMinDiameter(1)
dMinG = inputMinDiameter(0)
Di1 = max(Di1, dMinG, dMinH)
leftIRatio = Di2 / Di1
KfH, KfsH = inputStressConcentrationFactors(1, Di1, leftIRatio)
dMinH = inputSafetyFactors(1, Di1, KfH, KfsH)
dMinG = inputSafetyFactors(0, Di1, inputPoints[0]["concentrations"][2], inputPoints[0]["concentrations"][3])
if dMinH < dMinG:
    inputSafetyFactors(1, dMinG, KfH, KfsH)
inputShaftDiameters["Di1"] = max(Di1, dMinG, dMinH)

print("")
print("Points on the input shaft")
for p in inputPoints:
    print("Point", p["name"], ": Diameter:", p["d"], "Safety Factors (fatigue, yield): ", p["n"], "Von Mises Stresses (Alternating, Midrange):", p["VonMises"])
print(f"\nPoints on the output shaft")
for p in outputPoints:
    print("Point", p["name"], ": Diameter:", p["d"], "Safety Factors (fatigue, yield): ", p["n"], "Von Mises Stresses (Alternating, Midrange):", p["VonMises"])
print("")
print("Input Shaft Dimensions:")
print(inputShaftDiameters)
print("Output Shaft Dimensions:")
print(outputShaftDiameters)