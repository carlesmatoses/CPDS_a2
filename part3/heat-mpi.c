/*
 * Iterative solver for heat distribution
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

    int maxiter_tag = 0;
    int resolution_tag = 1;
    int algorithm_tag = 2;
    int u_tag = 3;
    int uhelp_tag = 4;
    int rows_tag = 5;
    double gresidual=0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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
        
        // starting time
        runtime = wtime();

        // send to workers the necessary data to perform computation
        int *rows_per_process = malloc(numprocs * sizeof(int));
        divide_rows(param.resolution, numprocs, rows_per_process);

        int row_offset = 1;
        for (int i=0; i<numprocs; i++) {
            if (i>0) {
                    MPI_Send(&param.maxiter, 1, MPI_INT, i, maxiter_tag, MPI_COMM_WORLD);
                    MPI_Send(&param.resolution, 1, MPI_INT, i, resolution_tag, MPI_COMM_WORLD);
                    MPI_Send(&rows_per_process[i], 1, MPI_INT, i, rows_tag, MPI_COMM_WORLD);
                    MPI_Send(&param.algorithm, 1, MPI_INT, i, algorithm_tag, MPI_COMM_WORLD);
                    MPI_Send(&param.u[(row_offset-1)*np], (rows_per_process[i]+2)*(np), MPI_DOUBLE, i, u_tag, MPI_COMM_WORLD);
                    MPI_Send(&param.uhelp[(row_offset-1)*np], (rows_per_process[i]+2)*(np), MPI_DOUBLE, i, uhelp_tag, MPI_COMM_WORLD);
            }
            row_offset += rows_per_process[i];
        }

        // allocate memory for root worker
        double * u = calloc( sizeof(double),(rows_per_process[0]+2)*(np) );
        double * uhelp = calloc( sizeof(double),(rows_per_process[0]+2)*(np) );
        if( (!u) || (!uhelp) )
        {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }
        // fill initial values for matrix with values of param.u and param.uhelp
        for (int i=0; i<rows_per_process[0]+2; i++)
            for (int j=0; j<np; j++) {
                u[ i*np+j ] = param.u[ i*np+j ];
                uhelp[ i*np+j ] = param.uhelp[ i*np+j ];
            }

        iter = 0;
        while(1) {
        switch( param.algorithm ) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, rows_per_process[0]+2, np);
                // Copy uhelp into u
                for (int i=0; i<rows_per_process[0]+2; i++)
                        for (int j=0; j<np; j++)
                        u[ i*np+j ] = uhelp[ i*np+j ];
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(param.u, np, np);
                break;
            case 2: // GAUSS
                residual = relax_gauss(param.u, np, np);
                break;
            }
            if (iter % 1000 == 0) {
                printf("iteration = %d\n with residual %f",  iter, gresidual);
            }
            iter++;

            if(numprocs > 1) {
                // send bottom row and recieve top row from worker 1
                MPI_Sendrecv(   &u[(rows_per_process[0]) * np], np, MPI_DOUBLE, 1, 0, 
                                &u[(rows_per_process[0] + 1) * np], np, MPI_DOUBLE, 1, 0, 
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            MPI_Reduce(&residual, &gresidual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&gresidual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // solution good enough ?
            if (gresidual < 0.00005) break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (param.maxiter>0 && iter>=param.maxiter) break;
        }

        // recieve the final matrix segment from each worker
        row_offset = rows_per_process[0]+1;
        for (int i=1; i<numprocs; i++) {
            MPI_Recv(&param.u[row_offset*np], rows_per_process[i]*np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            row_offset += rows_per_process[i];
        }
        // copy the u and uhelp into the first segment of the matrix
        for (int i=0; i<rows_per_process[0]+2; i++) {
            for (int j=0; j<np; j++) {
                param.u[i*np+j] = u[(i)*np+j];
                param.uhelp[i*np+j] = uhelp[(i)*np+j];
            }
        }


        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
            flop/1000000000.0,
            flop/runtime/1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

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

        printf("I am worker %d and ready to receive work to do ...\n", myid);

        // receive information from master to perform computation locally

        int columns, rows, np;
        int iter, maxiter;
        int algorithm;
        double residual;

        MPI_Recv(&maxiter,   1, MPI_INT, 0, maxiter_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&columns,   1, MPI_INT, 0, resolution_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows,      1, MPI_INT, 0, rows_tag, MPI_COMM_WORLD, &status);
        printf("Received rows = %d and columns = %d\n", rows, columns);
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
        MPI_Recv(&u[0], (rows+2)*(np), MPI_DOUBLE, 0, u_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&uhelp[0], (rows+2)*(np), MPI_DOUBLE, 0, uhelp_tag, MPI_COMM_WORLD, &status);

        iter = 0;
        while(1) {
        switch( algorithm ) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, rows+2, np);
                // Copy uhelp into u
                for (int i=1; i<rows+1; i++)
                    for (int j=0; j<np; j++)
                    u[ i*np+j ] = uhelp[ i*np+j ];
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np, np);
                break;
            case 2: // GAUSS
                residual = relax_gauss(u, np, np);
                break;
            }

            iter++;

            if (myid < numprocs - 1) {
                // Send top row to worker myid-1 and receive top ghost row from myid-1
                MPI_Sendrecv(&u[np], np, MPI_DOUBLE, myid - 1, 0,
                            &u[0], np, MPI_DOUBLE, myid - 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Send bottom row to worker myid+1 and receive bottom ghost row from myid+1
                MPI_Sendrecv(&u[rows * np], np, MPI_DOUBLE, myid + 1, 0,
                            &u[(rows + 1) * np], np, MPI_DOUBLE, myid + 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (myid == numprocs - 1) {
                // Send top row to worker myid-1 and receive top ghost row from myid-1
                MPI_Sendrecv(&u[np], np, MPI_DOUBLE, myid - 1, 0,
                            &u[0], np, MPI_DOUBLE, myid - 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            MPI_Reduce(&residual, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		    MPI_Bcast(&gresidual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // solution good enough ?
            if (gresidual < 0.00005) break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (maxiter>0 && iter>=maxiter) break;
        }

        // send the final matrix segment to the master
        MPI_Send(&u[np], (rows)*(np), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        if( u ) free(u); 
        if( uhelp ) free(uhelp);

        fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, gresidual);

        MPI_Finalize();
        exit(0);
    }
}
