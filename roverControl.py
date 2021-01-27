# Rover Control
def roverControl(n, m, commands):
    row, col, size = 0, 0, n
    for command in commands:
        if command == "RIGHT" and col < n:
            col += 1
        elif command == "LEFT" and col != 0:
            col -= 1
        elif command == "DOWN" and row < n:
            row += 1
        elif command == "UP" and row != 0:
            row -= 1
    return (row * size) + col
