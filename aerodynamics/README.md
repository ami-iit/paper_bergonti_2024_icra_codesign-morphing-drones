# Aerodynamics

This folder contains the aerodynamic models of the morphing drones. The models are stored in the `database_aerodynamic_models` folder. 
They are generated by solving a linear regression problem, implemented in [aero_model_id.py](../src/aero/aero_model_id.py), using the data stored in the `output` folder. The data in the `output` folder are generated and evaluated using the software `flow5`, employing the 3D uniform triangle panel Galerkin method.
For more details, please refer to section III.C of the paper.
