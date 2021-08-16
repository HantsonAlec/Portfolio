import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
Check image to see table for the rules
"""

# fuzzy vars
arrivals = ctrl.Antecedent(np.arange(1, 7, 1), "arrivals")
queue = ctrl.Antecedent(np.arange(1, 7, 1), "queue")
minutes = ctrl.Consequent(np.arange(0, 61, 1), "minutes")

# Membership functions
arrivals.automf(4, names=["AN", "F", "M", "TM"])
queue.automf(4, names=["VS", "S", "M", "L"])
minutes["Z"] = fuzz.trimf(minutes.universe, [0, 0, 0])
minutes["S"] = fuzz.trimf(minutes.universe, [0, 0, 30])
minutes["M"] = fuzz.trimf(minutes.universe, [0, 30, 60])
minutes["L"] = fuzz.trimf(minutes.universe, [30, 60, 60])

# Rules
rule1 = ctrl.Rule(arrivals["AN"], minutes["Z"])
rule2 = ctrl.Rule(arrivals["F"] & (queue["VS"] | queue["S"]), minutes["S"])
rule3 = ctrl.Rule(arrivals["F"] & (queue["M"] | queue["L"]), minutes["Z"])
rule4 = ctrl.Rule(arrivals["M"] & (queue["VS"] | queue["S"]), minutes["M"])
rule5 = ctrl.Rule(arrivals["M"] & queue["M"], minutes["S"])
rule6 = ctrl.Rule(arrivals["M"] & queue["L"], minutes["Z"])
rule7 = ctrl.Rule(arrivals["TM"] & (queue["VS"] | queue["L"]), minutes["L"])
rule8 = ctrl.Rule(arrivals["TM"] & (queue["S"] | queue["M"]), minutes["M"])


# Control system
minutesing_control = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
minutesing_simulator = ctrl.ControlSystemSimulation(minutesing_control)

# Inputs
minutesing_simulator.input['arrivals'] = 6
minutesing_simulator.input['queue'] = 6

# Compute answer and show
minutesing_simulator.compute()
print(f"{minutesing_simulator.output['minutes']} minutes")
minutes.view(sim=minutesing_simulator)
plt.show()
