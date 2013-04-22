README --
	Author: Eric Faurie
	Subject: CS 546 HW4
	Date: 21 APR 2013
	
Sections:
	Section 1: get_data.c
	Section 2: gauss_mpi.c
	Section 3: gauss_mpi2.c
	Section 4: gauss_mpi3.c
	
Section 1: get_data.c
	Source:
		The get_data.c file that has been modified to have the process
		with the largest rank be the collector.
	
	How to Compile:
		I have included a make file and the get_data.c file may be compiled
		by using the following command
		
			"make get_data"
		
		OR you may compile it manually with the following commands
		
			"mpicc -c get_data.c"
			"mpicc -o get_data get_data.c"
			
	How to Run:
		The makefile includes a run command that runs the file with two processes
		on a single machine, this may be executed using
		
			"make run_get_data"
		
		OR you may run the file manually with specified commands
		
			"mpirun -f MACHINE_FILE -np # ./get_data"
			
		Where -f MACHINE_FILE is the optional set of machines availible and # is
		the number of processes to execute.

Section 2: guass_mpi.c
	Source:
		A first attempt at the gaussian elimination problem in mpi
	
	Algorithm:
		The algorithm is incredibly naive. It only parallelizes each row operation.
		While this is fairly simple and it's easy to split up each row in the inner
		row (for) loop, it requires insane amounts of communication and therefore
		runs slower than the sequential algorithm, and, in fact, if N is large, it
		takes too long to complete. 
		It does function when N is small however.
		
		Note that this appraoch was taken largle because the matrix A[][] was non-
		contiguous, and therefore difficult to distribute.
	
	How to Compile:
		I have included a make file and the get_data.c file may be compiled
		by using the following command
		
			"make gauss_v1"
		
		It outputs the executable file gauss_mpi
		
	How to Run:
		Follow the standard run formatting specified in the project assignment.

Section 3: gauss_mpi2.c
	Source:
		A second attempt at the gaussian elimination problem in mpi
	
	Algorithm:
		The algorithm is much more sophisticated. It initially scatters all rows 
		and delivers them to the separate processes. Then each process executes an
		entire norm loop iteration on it's own. At the end of which, the process
		containing the most up to date normalization row broadcasts it to the other
		processes prior to restarting the next iteration. There is a bug somewhere in
		the program however, and it deadlocks when N is large. I did most testing where
		N = 5 and p = 2. If you run it like this with various print statements included
		you will notice that the result matrix A is correct and all elements of the 
		result matrix B are correct EXCEPT for the last index. I stared at it for hours
		and can't figure out why. So the answer it gives is incomplete. I'm assuming this
		same bug is what causes the deadlock when N is large.
		
		This algorithm is much improved over the previous one, however, it could still be
		optimized because as norm passes the local processors effective rows, the process
		remains idle waiting for the others who have not yet completed.
		
		Note too that the matrix A is now a contiguous matrix (this is neccessary)
	
	How to Compile:
		I have included a make file and the get_data.c file may be compiled
		by using the following command
		
			"make gauss_v2"
		
		It outputs the executable file gauss_mpi2
		
	How to Run:
		Follow the standard run formatting specified in the project assignment.
		
Section 4: gauss_mpi3.c
	Section 3: gauss_mpi2.c
	Source:
		A second attempt at the gaussian elimination problem in mpi
	
	Algorithm:
		Note this code is based loosley on code from an internet source.
		This algirithm is much like the pervious one, however it does many of the
		operations in place. The processes that complete early remain idle waiting
		for the later processes, the same problem with the algirithm above.
		
		This source file also does not work. It doesn't output the correct answer and
		it fails when N is large. I was simply trying to make something, anything,
		work properly. I have spent far too long on this without any sort of progress.
	
	How to Compile:
		I have included a make file and the get_data.c file may be compiled
		by using the following command
		
			"make gauss_v3"
		
		It outputs the executable file gauss_mpi3
		
	How to Run:
		Follow the standard run formatting specified in the project assignment.
		
Timing Results:
	For gauss_mpi the time goes up exponentially as N increses to the point where
	anything above 100 seems to take an endless amount of time, and therefore no time
	results were collected.
	
	For gauss_mpi2/3 the programs deadlock when N is large. So no time results were
	collected here either.