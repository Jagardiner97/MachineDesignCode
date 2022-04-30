import math
import matplotlib.pyplot as plt
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

points = [1.25, 1.75, 2.0, 3.75, 4.25, 5.25]
pointValues = []
n = 1.5
S_ut = 56_000 #psi
i = 81
while i <= 86:
    x_curr = points[i - 80]
    name = chr(i)
    moment = outputMoment(x_curr, maxShear, maxMoment)
    torque = outputTorqueDiagram(x_curr, outputShaftTorque)
    K_a = 2 * ((S_ut / 1000) ** -0.217)
    K_b = 0.9
    S_prime = S_ut / 2
    S_e = K_a * K_b * S_prime
    K_t = 2.7
    K_ts = 2.2
    K_f = 2.7
    K_fs = 2.2
    d_min = math.pow(16 * n / math.pi * (2 * K_f * moment) / S_e + math.pow(3 * math.pow(K_fs * torque, 2), 1/2) / S_ut, 1/3)
    dRatioMin = 1.1
    dRatioMax = 1.2
    currPoint = dict(name=chr(i), x=x_curr, moment=moment, torque=torque, K_t=K_t, K_ts=K_ts, K_f=K_f, K_fs=K_fs, K_a=K_a, K_b=K_b, S_prime=S_prime, S_e=S_e)
    pointValues.append(currPoint)
    i += 1
    Dmin = dRatioMin * d_min
    Dmax = dRatioMax * d_min
    print("")


print("")