# Simulation base for MA-STE-RL

## base.py
* Generate whole world(agent, gas, wind, wifi_link, etc.)
* Does not specify exact gas params
* In multi-agent case, communication connectivity checking and measurement sharing are conducted in base.py

## extreme.py
* Specify exact gas params
* In this environment, all parameters are randomly choosen = extreme case

## agent.py
* Containing some agent's own functions
* For example, gas_sensor, wind_sensor, ect.

## Requirement
Build CUDA image first (CUDA11.1 is tested)
