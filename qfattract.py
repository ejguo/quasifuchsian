

import numpy as np
import cmath
import matplotlib.pyplot as plt


def grandma(ta, tb, posroot=False, opt=0, tab=None):
    if tab is None:
        opt = opt % 2

    if opt < 2:  # Calculate the trace Tab from the grandfather identity
        p = -ta * tb
        q = ta**2 + tb**2

        if posroot:
            tab = (-p + cmath.sqrt(p**2 - 4 * q)) / 2
        else:
            tab = (-p - cmath.sqrt(p**2 - 4 * q)) / 2

        z0 = ((tab - 2) * tb) / (tb * tab - 2 * ta + 2j * tab)

        if opt == 0:  # Grandma's original recipe
            ab = [
                [tab / 2, (tab - 2) / (2 * z0)],
                [(tab + 2) * z0 / 2, tab / 2]
            ]

            b = [
                [(tb - 2j) / 2, tb / 2],
                [tb / 2, (tb + 2j) / 2]
            ]

            # Calculate matrix multiplication a = ab * b^-1
            det_b = b[0][0] * b[1][1] - b[0][1] * b[1][0]

            b_inv = [
                [b[1][1] / det_b, -b[0][1] / det_b],
                [-b[1][0] / det_b, b[0][0] / det_b]
            ]

            a = [
                [ab[0][0] * b_inv[0][0] + ab[0][1] * b_inv[1][0], ab[0][0] * b_inv[0][1] + ab[0][1] * b_inv[1][1]],
                [ab[1][0] * b_inv[0][0] + ab[1][1] * b_inv[1][0], ab[1][0] * b_inv[0][1] + ab[1][1] * b_inv[1][1]]
            ]

            return a, b
    return None, None

# Specify traces
ta = 3
tb = 3
a, b = grandma(ta, tb)
print("Matrix A:", a)
print("Matrix B:", b)


# Define the inverses of a and b
def inverse(matrix):
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return [[matrix[1][1] / det, -matrix[0][1] / det], [-matrix[1][0] / det, matrix[0][0] / det]]

a_inv = inverse(a)
b_inv = inverse(b)

# Define the DFS function
def generate_words_dfs(k):
    elements = [('a', 'a_inv'), ('b', 'b_inv')]
    words = set()

    def dfs(current_word, last_gen):
        if len(current_word.split()) <= k:
            words.add(current_word.strip())
            for gen, inv in elements:
                if last_gen != inv:
                    dfs(current_word + ' ' + gen, gen)
                if last_gen != gen:
                    dfs(current_word + ' ' + inv, inv)

    dfs('', None)
    return words

# Compute the matrix product, check the trace, and find the fixed points
def compute_matrix_products_and_fixed_points(words):
    word_to_matrix = {
        'a': a,
        'a_inv': a_inv,
        'b': b,
        'b_inv': b_inv
    }

    word_fixed_point_dict = {}

    for word in sorted(words):
        if word == '':
            product = [[1, 0], [0, 1]]  # Identity matrix for empty word
        else:
            matrices = [word_to_matrix[w] for w in word.split()]
            product = [[1, 0], [0, 1]]
            for matrix in matrices:
                product = [[product[0][0] * matrix[0][0] + product[0][1] * matrix[1][0],
                            product[0][0] * matrix[0][1] + product[0][1] * matrix[1][1]],
                           [product[1][0] * matrix[0][0] + product[1][1] * matrix[1][0],
                            product[1][0] * matrix[0][1] + product[1][1] * matrix[1][1]]]

        # Extract the matrix elements
        a_val = product[0][0]
        b_val = product[0][1]
        c_val = product[1][0]
        d_val = product[1][1]

        # Compute the discriminant for the fixed point formula
        discriminant = (d_val - a_val)**2 + 4 * b_val * c_val

        # Compute the fixed points
        if c_val != 0:  # To avoid division by zero
            fixed_point_1 = (a_val - d_val + cmath.sqrt(discriminant)) / (2 * c_val)
            fixed_point_2 = (a_val - d_val - cmath.sqrt(discriminant)) / (2 * c_val)

            # Derivative of the transformation at the fixed points
            def derivative(z):
                return (a_val * d_val - b_val * c_val) / (c_val * z + d_val)**2

            # Determine which is the attracting fixed point
            if abs(derivative(fixed_point_1)) < 1:
                attracting_fixed_point = fixed_point_1
            else:
                attracting_fixed_point = fixed_point_2

            word_fixed_point_dict[word] = attracting_fixed_point

    return word_fixed_point_dict

# Usage
k = 6
all_words = generate_words_dfs(k)
word_fixed_point_dict = compute_matrix_products_and_fixed_points(all_words)

# Evaluate fixed points to numerical values for plotting
attracting_fixed_points_evaluated = {word: (point.real, point.imag) for word, point in word_fixed_point_dict.items()}

from collections import OrderedDict

def find_ordered_fixed_points(word_fixed_point_dict):
    # Convert complex numbers to tuples of their real and imaginary parts
    evaluated_points = {word: (point.real, point.imag) for word, point in word_fixed_point_dict.items()}

    # Start with the fixed point corresponding to the word 'a'
    start_word = 'a'
    start_point = evaluated_points[start_word]

    ordered_dict = OrderedDict()
    ordered_dict[start_word] = start_point

    remaining_points = {word: point for word, point in evaluated_points.items() if word != start_word}

    current_point = start_point

    while remaining_points:
        # Find the next closest point to the current point
        next_word, next_point = min(remaining_points.items(),
                                    key=lambda item: np.hypot(item[1][0] - current_point[0], item[1][1] - current_point[1]))
        # Add the next closest point to the ordered dictionary
        ordered_dict[next_word] = next_point
        # Remove the selected point from the remaining points
        del remaining_points[next_word]
        # Update the current point to the newly selected point
        current_point = next_point

    return ordered_dict

# Usage
ordered_fixed_points = find_ordered_fixed_points(word_fixed_point_dict)


import matplotlib.pyplot as plt

# Colors
def generate_colors(num_points):
    colors = []
    r, g, b = 0, 0, 0
    for i in range(num_points):
        if i % 1 == 0:  # Change color every n points
            if r < 255 and g == 0 and b == 0:
                r = min(255, r + 1)
            elif r == 255 and b < 255 and g == 0:
                b = min(255, b + 1)
            elif r == 255 and b == 255 and g < 255:
                g = min(255, g + 1)
            elif b == 255 and r > 0:
                r = max(0, r - 1)
            elif r == 0 and g == 255 and b > 0:
                b = max(0, b - 1)
            elif g > 0 and r == 0 and b == 0:
                g = max(0, g - 1)
        colors.append((r / 255, g / 255, b / 255))  # Normalize to [0, 1] for matplotlib
    return colors

def plot_ordered_fixed_points(ordered_fixed_points, max_length):
    num_points = len(ordered_fixed_points)
    colors = generate_colors(num_points)

    plt.figure(figsize=(8, 8))

    # Plot fixed points using scatter
    for i, (x, y) in enumerate(ordered_fixed_points.values()):
        plt.scatter(x, y, color=colors[i], s=1)

    # Plot customization
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Fixed Points')
    plt.grid(True)

    plt.show()

    # Print words grouped by length
    print("Words grouped by length:")
    for length in range(2, max_length + 1):
        print(f"Words of length {length}:")
        for word in ordered_fixed_points.keys():
            if len(word.split()) == length:
                print(word)
        print()  # Add a blank line between groups

    # Print all words in order
    print("All words in order:")
    for word in ordered_fixed_points.keys():
        print(word)

# Set the value for `max_length`
n = 6  # Change this value to your desired maximum word length

# Usage
plot_ordered_fixed_points(ordered_fixed_points, n)
