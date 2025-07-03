# praise
Project Resilience Aquacrop Irrigation Strategy with Evolution

[Project Resilience](https://project-resilience.github.io/platform/) is a collaboration between the [Cognizant AI Lab](https://www.cognizant.com/us/en/ai-lab) and the United Nations ITU to use AI for sustainable development.

This project was created to demonstrate the use of evolutionary optimization to develop irrigation strategies for crops. Candidate evolved policies are evaluated using the AquaCrop model from the FAO. We use the [aquacropospy](https://github.com/aquacropos/aquacrop) library to run AquaCrop simulation in Python.

## Setup
This project was created using `python 3.10.18`.
Use `pip install -r full-requirements.txt` to install the required dependencies.

## Demo
A smaller version of the requirements necessary for the demo can be installed with `pip install -r requirements.txt`.
To run the demo use `streamlit run app/app.py`.
The demo is also hosted at `irrigation-strategy.streamlit.app`.