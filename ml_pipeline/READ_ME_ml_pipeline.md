# ML_Pipeline 

Machine Learning Algorithm to mimic what is done in data/Texas and/or data/Western. 

---

## Data
Data is very similar to what is used in Texas. However, instead of a .mat file, we use .csv.
Files:
    _00_case_texas.mat - original .mat file, used in _01_to_csv.py and load_demand, a function copied from data/Texas
    _01_branchTexas.csv to _04_gencostTexas.csv - csv parsing of _00_case_texas.mat, used in GNN and the form of data used in the majority of the directory
    _05_bus_network.png - example structure of network
    _06_texas_v0.csv - "ground truth" solution produced by data/Texas, used as output/target variables
A similar structure is shown with _11_branchWestern to _16_hosting_capacity_western.csv.

All csvs are made with _01_to_csv.py.

## Files
All files are as follows:
    _01_to_csv.py - transforms .mat file into the csvs
    _02_data_analysis.py - used to eliminate useless features with many zeros or empty columns
    _03_graph_construction.py - contains many functions, gives us input/output in structures we can pass through a GNN, along with function that checks physics_constraints, and our actual model used
    _04_longest_chain.py - used to check longest chain found in a grid network
    _05_gnn_trainer.py - contains trainer class used in _06_training.py, ported over from personal ML library
    _06_training.py - actually trains the GNN, uses rest of the files
    _07_test_results.py - checks any results, either from csv or model, against physics criterion

For more info, you can read the files yourself :) I'm not writing more of this readme right now

## List of Needed Additions
Progress needs to be made on:
    Checking if Physics Constraints are implemented correctly. Ground truth of _06_texas_v0.csv still violates physics constraints very slightly, by 0.02MW on average. Within margin of error, but it would be nice if we could check if these criteria are correct.
    Actually optimizing the model; I haven't started yet :D
    Fix .sh script, seemingly not running on CPU cluster
    Maybe a COPILOT.md/CLAUDE.md file


## Dependencies
```
numpy
pandas
matlibplot
torch
torch_geometric
```
