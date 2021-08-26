# Play for the Rich and Work for the Poor? 


Replication material to solve for Shanker and Wolfe (2021): "Play for the Rich and Work for the Poor?Optimal Saving and Work Hours in the Heterogeneous Agents Neoclassical Growth Model" [CEPR discussion paper DP16479](https://cepr.org/active/publications/discussion_papers/dp.php?dpno=16479)

Package solves the incomplete market equilibrium and constrained optima for a Bewley-Aiyagari-Huggett model with labour-leisure choice. The consumer problem is solved efficiently using the Endogenous Grid Method. 

Prefered useage is X-entropy using MPI

main_mpi.py solves IM and CP using cross entropy across multiple nodes. For example, to solve using N = 420 draws

``` mpiexec -n 420  python3 -m mpi4py main_mpi.py```

Note number of elite draws for cross entropy specified as N_elite paramter in main_mpi.py script. 



