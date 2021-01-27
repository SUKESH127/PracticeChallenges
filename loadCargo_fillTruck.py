# Load the Cargo / Fill the Truck
def loadTheCargo(num, containers, itemSize, itemsPerContainer, cargoSize):                    
    totalCargo = []
    n = num if num <= itemSize else itemSeize
    
    for i in range(n):
        listToAdd = [itemsPerContainer[i]]*containers[i]
        totalCargo += listToAdd
    
    totalCargo.sort(reverse=True)
    if cargoSize > len(totalCargo):
        cargoSize = len(totalCargo)

    outputSum = 0
    for i in range(cargoSize):
        first = totalCargo.pop(0)
        outputSum += first

    return outputSum
