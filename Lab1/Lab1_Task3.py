import heapq
import matplotlib.pyplot as plt
import time

# Define the grid with movement costs
grid = [
    [0, 2, 3],  # First row (Pac-Man starts at (0,0))
    [1, 5, 2],  # Second row
    [2, 1, 0]   # Third row (Apple is at (2,2))
]

# Define possible moves: (row_change, col_change)
moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

# Function to implement Uniform Cost Search (UCS)
def uniform_cost_search(grid, start, goal):
    rows, cols = len(grid), len(grid[0])  # Get grid size
    pq = []  # Priority queue (min-heap) for UCS
    heapq.heappush(pq, (0, start, [start]))  # Push (cost, position, path)
    visited = {}  # Dictionary to store the lowest cost to each cell

    while pq:
        cost, (r, c), path = heapq.heappop(pq)  # Get the lowest-cost path

        # If we reached the goal (Apple), return the path and total cost
        if (r, c) == goal:
            return path, cost

        # If we already visited this cell with a lower cost, skip it
        if (r, c) in visited and visited[(r, c)] <= cost:
            continue
        visited[(r, c)] = cost  # Mark the cell as visited with its cost

        # Explore all possible moves (right, down, left, up)
        for dr, dc in moves:
            nr, nc = r + dr, c + dc  # New row and column
            if 0 <= nr < rows and 0 <= nc < cols:  # Check boundaries
                new_cost = cost + grid[nr][nc]  # Calculate new cost
                heapq.heappush(pq, (new_cost, (nr, nc), path + [(nr, nc)]))  # Add to queue

    return None, float('inf')  # Return if no path is found

# Define start and goal positions
start = (0, 0)   # Pac-Man's start position
goal = (2, 2)    # Apple position

# Run UCS to find the best path
path, total_cost = uniform_cost_search(grid, start, goal)

# Visualization function to animate Pac-Man moving step by step
def visualize_path(grid, path, total_cost):
    rows, cols = len(grid), len(grid[0])  # Grid size
    fig, ax = plt.subplots()  # Create figure for visualization

    accumulated_cost = 0  # Tracks total cost at each step

    for step, (r, c) in enumerate(path):  # Iterate over path steps
        ax.clear()  # Clear previous frame

        # Draw the grid
        for i in range(rows):
            for j in range(cols):
                # Draw grid cell with blue border
                ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, edgecolor='blue', fill=False))
                # Display grid number (cost of moving into that cell)
                ax.text(j + 0.5, rows - 1 - i + 0.7, str(grid[i][j]), color='white', ha='center', va='center', fontsize=15)

        # Draw the Apple (Red Square)
        apple_x, apple_y = goal[1], rows - 1 - goal[0]
        ax.add_patch(plt.Rectangle((apple_x, apple_y), 1, 1, color='red'))  # Apple is a red square

        # Draw Pac-Man (Yellow Circle)
        pac_x, pac_y = c, rows - 1 - r  # Convert grid to plot coordinates
        ax.add_patch(plt.Circle((pac_x + 0.5, pac_y + 0.5), 0.3, color='yellow'))  # Pac-Man is a yellow circle

        # Update and display accumulated cost at each step
        if step > 0:  # First step has no movement cost
            prev_r, prev_c = path[step - 1]  # Previous position
            accumulated_cost += grid[r][c]  # Add cost for this step
            ax.text(pac_x + 0.5, pac_y - 0.2, f"{accumulated_cost}", color='white', ha='center', va='center', fontsize=12, fontweight='bold')

        # If it's the last step, display total cost on the Apple
        if (r, c) == goal:
            ax.text(apple_x + 0.5, apple_y + 0.5, f"Total: {total_cost}", color='black', ha='center', va='center', fontsize=15, fontweight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

        # Display step number
        ax.set_title(f"Step {step + 1}/{len(path)}")  # Show step count
        ax.set_xlim(0, cols)  # Set x-axis limits
        ax.set_ylim(0, rows)  # Set y-axis limits
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_facecolor('black')  # Set background color

        plt.pause(2)  # Pause for 2 seconds before next step

    plt.show()  # Show final frame

# Run the visualization
visualize_path(grid, path, total_cost)
