# Roll Dice
def rollDice(dice):
    def getAllDist(counts, cur): # counts = how often num shows, cur is number 1-7
        rotate = 0
        for i in range(1, 7):
            if i != cur:
                dist = calcDist(i, cur) # get distance for a specific side
                totalRotationsPerSide = counts[i] * dist 
                rotate += totalRotationsPerSide
        return rotate
    # get number of steps for each from/to combination
    def calcDist(fromInt, toInt):
        if fromInt == toInt:
            return 0
        return 2 if (fromInt + toInt == 7) else 1
    # stores how many of each side there
    counts = [0]*7 
    for dic in dice:
        counts[dic] += 1
    # find the minimum number of rotations 
    minimum = float('inf')
    for i in range(1, 7):
        minimum = min(minimum, getAllDist(counts, i))
    return minimum

