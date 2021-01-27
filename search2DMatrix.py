def search2DMatrix(matrix, target):
    # an empty matrix obviously does not contain `target` (make this check
    # because we want to cache `width` for efficiency's sake)
    if len(matrix) == 0 or len(matrix[0]) == 0: return False
    # cache these, as they won't change.
    height, width = len(matrix), len(matrix[0])
    # start our "pointer" in the bottom-left
    row, col = height - 1, 0
    while col < width and row >= 0:
        if matrix[row][col] > target:
            row -= 1
        elif matrix[row][col] < target:
            col += 1
        else: # found it
            return True
    return False
