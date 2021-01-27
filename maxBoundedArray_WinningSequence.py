# Maximum Bounded Array or Winning Sequence     
def maxBounded(num, lower, upper):
    diff = upper - lower
    # initial conditions check 
    if num >= (2 * (diff + 1)) or num < 3:
        return -1
    
    startingNumber = upper - 1
    while ((upper - startingNumber + 1) + diff) < num:
        startingNumber -= 1
    
    res = [0] * num
    i = 0
    while startingNumber < upper:
        res[i] = startingNumber
        startingNumber += 1
        i += 1
    
    res[i] = startingNumber
    startingNumber -= 1
    i += 1

    while i < num:
        res[i] = startingNumber
        startingNumber -= 1
        i += 1
    
    return res
