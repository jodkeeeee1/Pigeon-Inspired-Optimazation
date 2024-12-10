import numpy as np

# Hàm mục tiêu (tối thiểu hóa)
def objective_function(x):
    x = np.array(x)  # Đảm bảo x là mảng NumPy
    if x.size == 0:
        return np.array([])  # Trả về mảng rỗng nếu không có đầu vào
    return np.sum(x**2, axis=1)

# Khởi tạo tham số
def initialize_pigeons(num_pigeons, dim, lower_bound, upper_bound):
    positions = np.random.uniform(lower_bound, upper_bound, (num_pigeons, dim))
    return positions

# Toán tử Map and Compass
def map_and_compass_update(positions, velocities, global_best, R):
    num_pigeons = positions.shape[0]
    rand = np.random.rand(num_pigeons, positions.shape[1])
    velocities = velocities * np.exp(-R) + rand * (global_best - positions)
    positions += velocities
    return positions, velocities

# Toán tử Landmark
def landmark_update(positions):
    # Giảm số lượng chim bồ câu còn một nửa
    num_pigeons = max(positions.shape[0] // 2, 1)  # Đảm bảo ít nhất còn 1 bồ câu
    center = np.mean(positions[:num_pigeons], axis=0)
    rand = np.random.rand(num_pigeons, positions.shape[1])
    positions[:num_pigeons] += rand * (center - positions[:num_pigeons])
    return positions[:num_pigeons]

# Thuật toán PIO
def pigeon_inspired_optimization(objective_function, num_pigeons=30, dim=2, lower_bound=-10, upper_bound=10, max_iter=100):
    # Khởi tạo các tham số
    positions = initialize_pigeons(num_pigeons, dim, lower_bound, upper_bound)
    velocities = np.zeros_like(positions)
    global_best = positions[np.argmin(objective_function(positions))]
    R = 0.5  # Tham số suy giảm

    # Lặp qua các lần lặp
    for t in range(max_iter):
        if t < max_iter // 2:
            # Toán tử Map and Compass
            positions, velocities = map_and_compass_update(positions, velocities, global_best, R)
        else:
            # Toán tử Landmark
            positions = landmark_update(positions)

        # Đánh giá hàm mục tiêu
        fitness = objective_function(positions)
        current_best = positions[np.argmin(fitness)]
        if objective_function([current_best]) < objective_function([global_best]):
            global_best = current_best

        # In tiến trình
        print(f"Iteration {t+1}: Best Fitness = {objective_function([global_best])[0]}")

    return global_best, objective_function([global_best])[0]

# Chạy thuật toán
best_position, best_fitness = pigeon_inspired_optimization(objective_function)
print(f"\nBest Position: {best_position}")
print(f"Best Fitness: {best_fitness}")
