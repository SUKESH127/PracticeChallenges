# Treasure Island I
def treasureIslandI(matrix):
    def find_treasure(matrix, row, col, steps, minStep):
        rowCheckCondition = (row < 0 or row >= len(matrix))
        colCheckCondition = (col < 0 or col >= len(matrix[0]))
        if rowCheckCondition or colCheckCondition or matrix[row][col] == 'D' or matrix[row][col] == '#':
            return None, minStep
        elif matrix[row][col] == 'X':
            steps += 1
            if minStep > steps:
                minStep = steps
            return None, minStep
        else:
            tmp = matrix[row][col]
            matrix[row][col] = '#'
            steps += 1
            up = find_treasure(matrix, row-1, col, steps, minStep)
            down = find_treasure(matrix, row+1, col, steps, minStep)
            left = find_treasure(matrix, row, col-1, steps, minStep)
            right = find_treasure(matrix, row, col+1, steps, minStep)
            matrix[row][col] = tmp
            correctMove = min(left[1], right[1], up[1], down[1])
            return steps, correctMove
    return (find_treasure(matrix, 0, 0, -1, float('inf')))[1]
