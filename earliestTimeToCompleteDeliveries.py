# Earliest Time to complete deliveries
def earliestTime(numOfBuildings, buildingOpenTime, offloadTime):
    buildingOpenTime.sort()
    offloadTime.sort()
    ans = 0
    for i in range(0, len(offloadTime), 4):
        time = buildingOpenTime[len(buildingOpenTime) - floor(i/4) - 1] + offloadTime[i + 3]
        ans = max(ans, time)
    return ans
