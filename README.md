# Identification of Braess links in traffic networks

> Semester project in [HOMES](https://www.epfl.ch/labs/homes/) laboratory at EPFL.
> 
> Author: Julien ARS in autumn 2024, under the supervision of Dr. Hossein R. Farahani and Prof. Kenan Zhang.

This repo countains the code and outputs of a semester project I did at EPFL in Homes laboratory.

In order to run the code, it is needed to download the transportation networks from [here](https://github.com/bstabler/TransportationNetworks), and to put them in a `TransportationNetworks` folder (or at least the SiouxFalls subfolder).

The files are organised as follows:

## Final report
The final report can be found in [Identification_of_Braess_links.pdf](Identification_of_Braess_links.pdf).

## Algorithms
The implemented algorithms can be found in the following files :
- [FrankWolf.py](FrankWolf.py) : Frank Wolf, including shortest path algorithm and cost functions.
- [EntropyMaximisation.py](EntropyMaximisation.py) : EMARB, including backward and forward entropy maximisa-
tion.
- [RemoveBraess.py](EntropyMaximisation.py) : The removal of links from the network.
- [Network.py](Network.py) : The implementation of the Network class, which supersedes the Graph class
from graph-tool [3], and implements loading a network out of the TransportationNetworks
files as well as drawing, saving, exporting and loading routines.

## Notebooks

- [Initial network calculations.ipynb](Initial%20network%20calculations.ipynb) Countains the code for the inital network calculations, such as solving SiouxFalls at UE and SO.
- [hypothesis testing.ipynb](hypothesis%20testing.ipynb) Countains the code for validating our hypothesis and analyse the results.
- [example network.ipynb](example%20network.ipynb) Apply our code on the example network in Xie and Nie's *A new algorithm for achieving proportionality in user equi-
librium traffic assignment.*

## Outputs

- The [exports](exports) folder countains our results in a human readable format.
- The [files](files) folder countains the results in a machine readable format.
- The [figs](figs) folder countains the figures for the final report.
