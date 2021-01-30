import matplotlib.pyplot as plt
import pandas as pd

DELTA_PLT_X = 1
DELTA_PLT_Y = 1

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }

# DATA
df = pd.read_csv('score.csv')

x_values = []
y_values = []

for i, row in df.iterrows():
    x_values.append(int(row.values[1]))
    y_values.append(float(row.values[2][7:-1]))

# FIGURE
plt.style.use('default')

hl, = plt.plot([], [])

plt.xlabel('Time (FL rounds)', fontdict=font, labelpad=5)
plt.ylabel('Prediction', fontdict=font, labelpad=5)
plt.title('Covert Channel Communication \n via Score Attack to a FL model', fontdict=font)
plt.plot(x_values, y_values, linestyle='--', marker = 'x', color='black', label='Text')

# Adjust axes
y_min = min(y_values) - DELTA_PLT_Y
y_max = max(y_values) + DELTA_PLT_Y
plt.ylim(y_min, y_max)

x_min = min(x_values) - DELTA_PLT_X
x_max = max(x_values) + DELTA_PLT_X
plt.xlim(x_min, x_max)

# Customize the plot
plt.grid(1, ls='--', color='#777777', alpha=0.5, lw=1)
plt.tick_params(labelsize=12, length=0)

# add a legend
leg = plt.legend( fontsize = 16, loc=1 )
fr = leg.get_frame()
fr.set_facecolor('w')
fr.set_alpha(.7)


plt.draw()
plt.show()

# SAVE
plt.savefig('score-output.png', dpi=300)
plt.savefig('score-output.svg', dpi=300)
