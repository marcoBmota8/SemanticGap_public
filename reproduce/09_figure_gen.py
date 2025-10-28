# This is code to generate publication ready figures
import matplotlib.pyplot as plt
import numpy as np

from src.directories import FIG_DIR

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Define range from left contact point to right contact point
x = np.linspace(0.0, 1.0, 300)

ypareto = np.sqrt(1-(x-0.05)**3)*0.99+0.01
ypareto2 = -0.7*x+0.98
ypareto3 = 1/((4*x+1.05))

# Plot
fig = plt.figure(figsize=(4.2,4.2))
plt.plot(x, ypareto, color='gray', lw=2.5, linestyle='--', alpha=0.5, label='Problem-specific\nPareto frontier') # Concave (Rare): Causes are very accurate (e.g. newborn genetic screening in cystic fibrosis) but adding some effects gives an extra boost in accuracy (adulthood Cl sweat test)
plt.plot(x, ypareto2, color='gray', lw=2.5, linestyle='--', alpha=0.5) # Linear tradeoff between reasoning from effects and from causes that are about equally accurate (e.g. type 2 diabetes Dx: C{BMI, age, family history of diabetes, underlying cardiovascular disease diagnosis} & E{A1c, fasting glucose levels)
plt.plot(x, ypareto3, color='gray', lw=2.5, linestyle='--', alpha=0.5) #Convex (Most common): Reasoning from highly accurate effects (symptoms, lab values, vital signs, etc)
plt.vlines(x=1.0, ymin=0.0, ymax=1.0, color='red', lw=1.5, linestyle='--')
plt.hlines(y=1.0, xmin=0.0, xmax=1.0, color='red', lw=1.5, linestyle='--')
# Fill the lens area
plt.fill_between(x, ypareto, 0.0, color='skyblue', alpha=0.25)
plt.fill_between(x, ypareto2, 0.0, color='skyblue', alpha=0.45)
plt.fill_between(x, ypareto3, 0.0, color='skyblue', alpha=0.9)

# Example points (toy models)
points = np.array([
    [0.03, 0.765],
    [0.5, 0.635],
    [0.85, 0.35],
    [0.85, 0.7],
    [0.5, 0.54],
    [0.15, 0.3]

])

dot_labels = ['Late\nretrospective', 'Early\n retrospective', 'Pre-Dx', 'Monogenic\nprimary\nscreening', 'Simple\nPRS', 'Complex\nPRS']
pos = [(0.02, 0.8), (0.1, 0.6), (0.7, 0.4), (0.65, 0.74), (0.45, 0.38), (0.1, 0.34)]

plt.scatter(points[:,0], points[:,1], s=80, color='blue', edgecolor='black', zorder=5)
for i, (px, py) in enumerate(points):
    plt.text(*pos[i], dot_labels[i], fontsize=16)

# Ideal case: cross at (1.0, 1.0)
plt.scatter([1.0], [1.0], marker='x', color='red', s=150, label='Ideal case', linewidths=3)

# Axes and formatting
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('Etiological Concordance\n$\mathit{[Anc(D) \equiv Anc(L)]}$', fontsize=16)
plt.ylabel('$\mathit{[D = L]}$\nValue  Concordance', fontsize=16)
plt.xlim(-0.01, 1.05)
plt.ylim(-0.01, 1.05)
plt.grid(alpha=0.4)
plt.legend(fontsize=16, bbox_to_anchor=(0.97, -0.02), loc='lower right')
plt.gca().set_aspect('equal')

plt.show()

fig.savefig(FIG_DIR/'pareto_acc.pdf', dpi=300, bbox_inches='tight')
