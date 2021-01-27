# Utilization Checks
def finalInstances(instances, averageUtil):
    end = len(averageUtil)
    i = 0
    lim = 2 * (10**8)
    while i < end:
        if averageUtil[i] < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            i += 10
        elif averageUtil[i] > 60 and instances < lim:
            instances *= 2 
            i += 10
        i += 1
    return instances
