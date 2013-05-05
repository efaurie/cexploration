Author: Eric Faurie
Class: CS 546 - Project
Date: 05 MAR 2013

Compilation:
	
	Background:
		I have included a makefile to aid in the compilation of the source files.
	
	Compile All Source:
		To compile all the source files, 2D-Conv.c, 2D-Conv-P1.c, 2D-Conv-P2.c, and 2D-Conv-P3.c
		enter the command:
			"make"
		
		It will generate the executables 2D-Conv, 2D-Conv-P1, 2D-Conv-P2, and 2D-Conv-P3
	
	Compile 2D-Conv:
		To compile only the 2D-Conv.c file enter the command:
			"make 2D-Conv"
		
		It will generate the executable 2D-Conv
	
	Compile 2D-Conv-P1:
		To compile only the 2D-Conv-P1.c file enter the command:
			"make 2D-Conv-P1"
		
		It will generate the executable 2D-Conv-P1
		
	Compile 2D-Conv-P2:
		To compile only the 2D-Conv-P2.c file enter the command:
			"make 2D-Conv-P2"
		
		It will generate the executable 2D-Conv-P2
		
	Compile 2D-Conv-P3:
		To compile only the 2D-Conv-P3.c file enter the command:
			"make 2D-Conv-P3"
		
		It will generate the executable 2D-Conv-P3
	

	
Running:

	Background:
		This section assumes the files required have been compiled.

	Running 2D-Conv:
		To run 2D-Conv (the serial algorithm) please enter the command:
			"./2D-Conv"
	
	Running 2D-Conv-P1:
		To run 2D-Conv-P1 (the Send/Recv algorithm) please enter the command:
			"mpirun -np # ./2D-Conv-P1"    where # is the number of processes
			
	Running 2D-Conv-P2:
		To run 2D-Conv-P2 (the Scatter/Gather algorithm) please enter the command:
			"mpirun -np # ./2D-Conv-P2"    where # is the number of processes
			
	Running 2D-Conv-P3:
		To run 2D-Conv-P3 (the Multi-Comm algorithm) please enter the command:
			"mpirun -np # ./2D-Conv-P3"    where # is the number of processes