import numpy as np
import matplotlib.pyplot as plt

# Ορισμός των συναρτήσεων συμμετοχής
def mu_A(x):
    return np.where(x > 15, 1 / (1 + (x - 15)**-2), 0)

def mu_B(x):
    return 1 / (1 + (x - 17)**4)

def mu_C(x):
    return np.where(x <= 15, 1, 0)

# Ορισμός των τελεστών για τα τρία ερωτήματα
def fuzzy_and_min(mu1, mu2):
    return np.minimum(mu1, mu2)

def fuzzy_and_prod(mu1, mu2):
    return mu1 * mu2

def fuzzy_and_bounded(mu1, mu2):
    return np.maximum(0, mu1 + mu2 - 1)

def fuzzy_or_max(mu1, mu2):
    return np.maximum(mu1, mu2)

def fuzzy_or_prob(mu1, mu2):
    return mu1 + mu2 - mu1 * mu2

def fuzzy_or_bounded(mu1, mu2):
    return np.minimum(1, mu1 + mu2)

# Δημιουργία του διαστήματος τιμών x
x = np.linspace(10, 25, 400)

# Υπολογισμοί για το C(x)
mu_A_values = mu_A(x)
mu_B_values = mu_B(x)
C_min = fuzzy_and_min(mu_A_values, mu_B_values)
C_prod = fuzzy_and_prod(mu_A_values, mu_B_values)
C_bounded = fuzzy_and_bounded(mu_A_values, mu_B_values)

# Υπολογισμοί για το D(x)
D_max = fuzzy_or_max(mu_A_values, mu_B_values)
D_prob = fuzzy_or_prob(mu_A_values, mu_B_values)
D_bounded = fuzzy_or_bounded(mu_A_values, mu_B_values)

# Υπολογισμοί για το E(x) με την απλή μορφή του μ_C(x)
mu_C_values = mu_C(x)
E_min = fuzzy_and_min(mu_C_values, mu_B_values)
E_prod = fuzzy_and_prod(mu_C_values, mu_B_values)
E_bounded = fuzzy_and_bounded(mu_C_values, mu_B_values)

# Σχεδίαση όλων των συναρτήσεων
plt.figure(figsize=(18, 10))
plt.subplot(3, 1, 1)
plt.plot(x, mu_A_values, label='μ_A(x)', linestyle='--')
plt.plot(x, mu_B_values, label='μ_B(x)', linestyle='--')
plt.plot(x, C_min, label='C_min (Minimum)')
plt.plot(x, C_prod, label='C_prod (Product)')
plt.plot(x, C_bounded, label='C_bounded (Bounded Difference)')
plt.title('Membership Functions for C(x)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, mu_A_values, label='μ_A(x)', linestyle='--')
plt.plot(x, mu_B_values, label='μ_B(x)', linestyle='--')
plt.plot(x, D_max, label='D_max (Maximum)')
plt.plot(x, D_prob, label='D_prob (Probabilistic Sum)')
plt.plot(x, D_bounded, label='D_bounded (Bounded Sum)')
plt.title('Membership Functions for D(x)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, mu_C_values, label='μ_C(x)', linestyle='--')
plt.plot(x, mu_B_values, label='μ_B(x)', linestyle='--')
plt.plot(x, E_min, label='E_min (Minimum)')
plt.plot(x, E_prod, label='E_prod (Product)')
plt.plot(x, E_bounded, label='E_bounded (Bounded Difference)')
plt.title('Membership Functions for E(x)')
plt.legend()

plt.xlabel('x')
plt.tight_layout()
plt.show()
