import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Fuzzy vars
quality = ctrl.Antecedent(np.arange(1, 11, 1), "quality")
food = ctrl.Antecedent(np.arange(1, 11, 1), "food")
tip = ctrl.Consequent(np.arange(0, 26, 1), "tip")

# Membership functions
quality.automf(3, names=["Poor", "Acceptable", "Amazing"])
food.automf(3, names=["Bad", "Decent", "Great"])


# Manual chart generation
tip["Low"] = fuzz.trimf(tip.universe, [0, 0, 13])
tip["Medium"] = fuzz.trimf(tip.universe, [0, 13, 25])
tip["High"] = fuzz.trimf(tip.universe, [13, 25, 25])


# Rules
rule1 = ctrl.Rule(quality["Amazing"] | food["Great"], tip["High"])
rule2 = ctrl.Rule(quality["Acceptable"], tip["Medium"])
rule3 = ctrl.Rule(quality["Poor"] | food["Bad"], tip["Low"])

# Control system
tipping_control = ctrl.ControlSystem([rule1, rule2, rule3])
tipping_simulator = ctrl.ControlSystemSimulation(tipping_control)

# Inputs
tipping_simulator.input['quality'] = 6.5
tipping_simulator.input['food'] = 9.8

# Compute answer and show
tipping_simulator.compute()
tip.view(sim=tipping_simulator)
plt.show()
