# Slowest Key
def slowestKey(keyTimes):
    longest, prev = ['',0], ['',0] # [character, time]
    keysPressed = [chr(keyTimes[i][0] + 97) for i in range(len(keyTimes))]
    for i, c in enumerate(keysPressed):
        duration = keyTimes[i][1] - prev[1] # calculate duration each time
        if duration > longest[1]:
            longest[0], longest[1] = c, duration
        elif duration == longest[1]: # break ties
            if c > prev[0]:
                longest[0], longest[1] = c, duration
            else:
                longest[0], longest[1] = prev[0], prev[1]
        prev = [c, keyTimes[i][1]] # set the previous character and time
    return longest[0]
