# Schedule Tasks
def processLogFile(logs, threshold):
    transactionMap = {}
    for i in range(len(logs)):
        logArray = logs[i].split()
        idOne = logArray[0]
        idTwo = logArray[1]
        if idOne == idTwo:
            if idOne in transactionMap:
                transactionMap[idOne] += 1
            else:
                transactionMap[idOne] = 1
        else:
            if idOne in transactionMap:
                transactionMap[idOne] += 1
            else:
                transactionMap[idOne] = 1
            
            if idTwo in transactionMap:
                transactionMap[idTwo] += 1
            else:
                transactionMap[idTwo] = 1
    
    output = []       
    ids = transactionMap.keys()
    for ID in ids:
        if transactionMap[ID] >= threshold:
            output.append(ID)
    return output     
