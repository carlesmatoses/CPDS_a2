/*
 * Iterative solver for heat distribution
 */

/*
The code is separated in two parts:
1. root:
    The root is excecuted when myid == 0. It reads the input file, initializes the parameters.
    - calculate the number of rows per process with divide_rows()
    - send with MPI_Send the divided matrix num_rows*np to each worker
    - wait for the results from the workers
    - gather the final results from the workers 
    - decide if the solution is good enough or the max iterations are reached
    - if good: send a stop signal to all workers

2. workers
    - wait for the matrix and parameters from the root.
    - if the stop signal is received, send the final results to the root
    - calculate the residual and send it to the root


*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

void divide_rows(int total_rows, int num_processes, int *rows_per_process) {
    int q = total_rows / num_processes;  // Integer division (this is 32 in your case)
    int r = total_rows % num_processes;  // Remainder (this is 2 in your case)

    for (int i = 0; i < num_processes; i++) {
        if (i < r) {
            rows_per_process[i] = q + 1;  // The first 'r' processes get an extra row
        } else {
            rows_per_process[i] = q;      // The remaining processes get 'q' rows
        }
    }
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    // tags:
    int maxiter_tag = 0;
    int resolution_tag = 1;
    int algorithm_tag = 2;
    int u_tag = 3;
    int uhelp_tag = 4;
    double gresidual=0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
    if (myid == 0) {
      printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;

    // check arguments
    if( argc < 2 )
    {
	usage( argv[0] );
	return 1;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) 
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
      
	usage(argv[0]);
	return 1;
    }

    // check result file
    resfilename= (argc>=3) ? argv[2]:"heat.ppm";

    if( !(resfile=fopen(resfilename, "w")) )
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for writing.\n\n", 
		resfilename);
	usage(argv[0]);
	return 1;
    }

    // check input
    if( !read_input(infile, &param) )
    {
	fprintf(stderr, "\nError: Error parsing input file.\n\n");
	usage(argv[0]);
	return 1;
    }
    print_params(&param);

    // set the visualization resolution
    
    param.u     = 0;
    param.uhelp = 0;
    param.uvis  = 0;
    param.visres = param.resolution;
   
    if( !initialize(&param) )
	{
	    fprintf(stderr, "Error in Solver initialization.\n\n");
	    usage(argv[0]);
            return 1;
	}

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;

    // calculate the number of rows per process with divide_rows()
    int *rows_per_process = malloc(numprocs * sizeof(int));
    divide_rows(np, numprocs, rows_per_process);
    printf("Rows per process: ");
    for (int i = 0; i < numprocs; i++) {
        printf("%d ", rows_per_process[i]);
    }
    printf("\n");
    // starting time
    runtime = wtime();

    // send to workers the necessary data to perform computation
    int row_offset = 0;
    for (int i=0; i<numprocs; i++) {
        if (i>0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, maxiter_tag, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, resolution_tag, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, algorithm_tag, MPI_COMM_WORLD);
                MPI_Send(&param.u[row_offset*np], (rows_per_process[i])*(np), MPI_DOUBLE, i, u_tag, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[row_offset*np], (rows_per_process[i])*(np), MPI_DOUBLE, i, uhelp_tag, MPI_COMM_WORLD);
        }
        row_offset += rows_per_process[i];
    }

    iter = 0;
    while(1) {
	switch( param.algorithm ) {
        case 0: // JACOBI
            residual = relax_jacobi(param.u, param.uhelp, rows_per_process[0]+1, np);

            // Copy uhelp into u (dont copy the ghost rows)
            for (int i=1; i<rows_per_process[0]; i++)
                for (int j=0; j<np; j++)
                    param.u[ i*np+j ] = param.uhelp[ i*np+j ];

            MPI_Reduce(&residual, &gresidual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&gresidual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            // Send the last row of the first worker to the second worker
            if (numprocs > 1) {
                MPI_Sendrecv(&param.u[(rows_per_process[0]-1) * np], np, MPI_DOUBLE, 1, 0,
                    &param.u[(rows_per_process[0]) * np], np, MPI_DOUBLE, 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;

	    case 1: // RED-BLACK
		    residual = relax_redblack(param.u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(param.u, np, np);
		    break;
	    }

        iter++;

        // solution good enough ?
        if (gresidual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }

    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;

    // recollect the final results from the workers
    int offset = rows_per_process[0] * np;  // Start offset for the first worker's data

    for (int i = 1; i < numprocs; i++) {
        int rows = rows_per_process[i];
        MPI_Recv(&param.u[offset], rows * np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&param.uhelp[offset], rows * np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += rows * np;
    }

    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f ", runtime);
    fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
	    flop/1000000000.0,
	    flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", gresidual, iter);

    // for plot...
    coarsen( param.u, np, np,
	     param.uvis, param.visres+2, param.visres+2 );
  
    write_image( resfile, param.uvis,  
		 param.visres+2, 
		 param.visres+2 );

    finalize( &param );

    fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, gresidual);

    MPI_Finalize();

    return 0;

} else {
// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
    printf("I am worker %d and ready to receive work to do ...\n", myid);

    // receive information from master to perform computation locally

    int columns, rows, np;
    int iter, maxiter;
    int algorithm;
    double residual;
    int *rows_per_process = malloc(numprocs * sizeof(int));
    
    MPI_Recv(&maxiter,   1, MPI_INT, 0, maxiter_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&columns,   1, MPI_INT, 0, resolution_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&algorithm, 1, MPI_INT, 0, algorithm_tag, MPI_COMM_WORLD, &status);

    divide_rows(columns+2, numprocs, rows_per_process);

    rows = rows_per_process[myid];
    np = columns + 2;

    // allocate memory for worker
    double * u = calloc( sizeof(double),(rows+2)*(np) );
    double * uhelp = calloc( sizeof(double),(rows+2)*(np) );
    if( (!u) || (!uhelp) )
    {
        fprintf(stderr, "Error: Cannot allocate memory\n");
        return 0;
    }
    
    // fill initial values for matrix with values received from master
    MPI_Recv(&u[np], (rows)*(np), MPI_DOUBLE, 0, u_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&uhelp[np], (rows)*(np), MPI_DOUBLE, 0, uhelp_tag, MPI_COMM_WORLD, &status);

    iter = 0;
    while(1) {
	switch( algorithm ) {
	    case 0: // JACOBI
            residual = relax_jacobi(u, uhelp, rows+2, np);
		    // Copy uhelp into u
		    for (int i=1; i<rows+1; i++)
                for (int j=0; j<np; j++)
	    		    u[ i*np+j ] = uhelp[ i*np+j ];

            MPI_Reduce(&residual, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		    MPI_Bcast(&gresidual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (myid < numprocs - 1) {
                // Send top row to worker myid-1 and receive top ghost row from myid-1
                MPI_Sendrecv(&u[np], np, MPI_DOUBLE, myid - 1, 0,
                            &u[0], np, MPI_DOUBLE, myid - 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Send bottom row to worker myid+1 and receive bottom ghost row from myid+1
                MPI_Sendrecv(&u[rows_per_process[myid] * np], np, MPI_DOUBLE, myid + 1, 0,
                            &u[(rows_per_process[myid] + 1) * np], np, MPI_DOUBLE, myid + 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (myid == numprocs - 1) {
                // Send top row to worker myid-1 and receive top ghost row from myid-1
                MPI_Sendrecv(&u[np], np, MPI_DOUBLE, myid - 1, 0,
                            &u[0], np, MPI_DOUBLE, myid - 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(u, np, np);
		    break;
	    }

        iter++;

        // solution good enough ?
        if (gresidual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (maxiter>0 && iter>=maxiter) break;
    }
    // send the final results to the master
    MPI_Send(&u[np], rows_per_process[myid] * np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&uhelp[np], rows_per_process[myid] * np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    if( u ) free(u); 
    if( uhelp ) free(uhelp);

    fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, gresidual);

    MPI_Finalize();
    exit(0);
  }
}
