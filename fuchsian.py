import sympy as sp
from sympy import I
import matplotlib.pyplot as plt

# Define the variables
u = 3
v = 3
w = 3
# w = (u * v + sp.sqrt(u**2 * v**2 - 4*(u**2 + v**2))) / 2

# Jorgensen parameters
a = sp.Matrix([[u - v / w, v / w**2], [u, v / w]])
b = sp.Matrix([[v - u / w, -v / w**2], [-v, u / w]])

# Button parameters
# a = sp.Matrix([[(1 + v**2) / w, v], [v, w]])
# b = sp.Matrix([[(1 + w**2) / v, -w], [-w, v]])

# Define the inverses of a and b
a_inv = a.inv()
b_inv = b.inv()

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

    fixed_points = []
    for word in sorted(words):
        if word == '':
            product = sp.eye(2)  # Identity matrix for empty word
        else:
            matrices = [word_to_matrix[w] for w in word.split()]
            product = sp.eye(2)
            for matrix in matrices:
                product = product * matrix

        # print(word, product)

        # Check the trace
        trace = product.trace()
        if abs(trace) <= 1.98:
            print(f"Error: trace <= 2 for word '{word}'")

        # Extract the matrix elements for real values
        a_val = round(product[0, 0], 3)
        b_val = round(product[0, 1], 3)
        c_val = round(product[1, 0], 3)
        d_val = round(product[1, 1], 3)

        ''' # Extract the matrix elements
        a_val = product[0, 0]
        b_val = product[0, 1]
        c_val = product[1, 0]
        d_val = product[1, 1] '''

        # Compute the discriminant for the fixed point formula
        discriminant = (d_val - a_val)**2 + 4 * b_val * c_val

        # Check for computation errors
        if discriminant < 0:
          print(word,product, discriminant, a_val, b_val, c_val, d_val)

        # Compute the fixed points
        if c_val != 0:  # To avoid division by zero
            fixed_point_1 = (a_val - d_val + sp.sqrt(discriminant)) / (2 * c_val)
            fixed_point_2 = (a_val - d_val - sp.sqrt(discriminant)) / (2 * c_val)
            fixed_points.append(fixed_point_1)
            fixed_points.append(fixed_point_2)

    return fixed_points

# Iterations
k = 10
all_words = generate_words_dfs(k)
fixed_points = compute_matrix_products_and_fixed_points(all_words)

# Evaluate fixed points to numerical values for plotting
fixed_points_evaluated = [(sp.re(point.evalf()), sp.im(point.evalf())) for point in fixed_points]

# Plotting the fixed points
x_vals = [x for x, y in fixed_points_evaluated]
y_vals = [y for x, y in fixed_points_evaluated]

plt.figure(figsize=(15, 15))
plt.scatter(x_vals, y_vals, color='blue', s=2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Fixed Points')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()
print(a,b)
