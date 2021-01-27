# Two Sum
def twoSum(self, nums, target):
    seen = {}
    for i, x in enumerate(nums):
        otherNumInPair = target - x
        if x not in seen:
            seen[x] = i
        if otherNumInPair in seen:
            output = [seen[otherNumInPair], i]
    return output
