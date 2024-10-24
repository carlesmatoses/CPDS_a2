#include <mpi.h>
#include "heat.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 16
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    for (int i = 1; i <= sizex - 2; i++) {
        for (int j = 1; j <= sizey - 2; j++) {
            utmp[i * sizey + j] = 0.25 * (
                u[i * sizey + (j - 1)] +  // left
                u[i * sizey + (j + 1)] +  // right
                u[(i - 1) * sizey + j] +  // top
                u[(i + 1) * sizey + j]    // bottom
            );
            diff = utmp[i * sizey + j] - u[i * sizey + j];
            sum += diff * diff;
        }
    }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nby, by, numprocs, myid, previd, nextid;

    MPI_Status status[NB];
    MPI_Request req[NB];

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    previd = myid == 0 ? MPI_PROC_NULL : myid - 1;
    nextid = myid == numprocs - 1 ? MPI_PROC_NULL : myid + 1;

    nby = NB;
    by = sizey/nby;

    for (int jj=0; jj<nby; jj++) {

        // Read first line from prev process
        MPI_Recv(u, by, MPI_DOUBLE, previd, jj, MPI_COMM_WORLD, status);

        for (int i=1; i<=sizex-2; i++) {
            for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                                  u[ i*sizey	+ (j+1) ]+  // right
                                  u[ (i-1)*sizey	+ j     ]+  // top
                                  u[ (i+1)*sizey	+ j     ]); // bottom
                diff = unew - u[i*sizey+ j];
                sum += diff * diff;
                u[i*sizey+j]=unew;
            }
        }

        // Send last line to next process
        MPI_Isend(&u[(sizex-2) * sizey + jj * sizex], by, MPI_DOUBLE, nextid, jj, MPI_COMM_WORLD, &req[jj]);
    }

    MPI_Waitall(nby, req, status);

    return sum;
}

