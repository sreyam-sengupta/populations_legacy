# Populations
Efficacy of active regulation vs passive bet-hedging in colonies of single-celled organisms.

## Files
*0_populations_basic.ipynb* is the basic file containing a working solver for the nonlinear 1D FP equation that is our toy model.

*1_populations_fitness_function.ipynb* generates a fitness contour map.

 *2a_populations_static_env_precise_sensing.ipynb* calculates fitness and information phase diagrams for various parameter pairs (alpha, delta) over ten different equally-spaced static sugar environments between 60 to 430 microM. Two phase diagrams showing the benefit conferred by regulation (post-selection) compared to selection without regulation are generated. The fitness conferred per bit of information in the post-selection case with regulation is calculated and a phase diagram generated.
 
 *2b_populations_static_env_imprecise_sensing.ipynb* explores imprecise sensing. The perceived sugar is distributed as a Gaussian with true value as the mean and s.d. given by sensing error * mean. From this pdf, and the inverse of the function from sugar to optimal expression level given sugar, we can construct a pdf of expression level, whose mean is the 'perceived optimal' expression level the cells think they should express at. The fitness and information phase diagrams are calculated for fifteen pairs of (alpha, sigma) and for each, the colony fitness is calculated for ten sigar environments ranging between 80 to 400 microM. These fitnesses are averaged and a phase diagram is generated.
