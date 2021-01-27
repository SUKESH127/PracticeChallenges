# Unique Device Names
def uniqueDeviceNames(num, deviceNames):
    deviceMap = {}
    output = []
    for i, device in enumerate(deviceNames):
        newString = device
        if not device in deviceMap:
            deviceMap[device] = 1
            output.append(device)
        else:
            newString += str(deviceMap[device])
            output.append(newString)
            deviceMap[device] += 1
    return output 
