#include "heat.h"
#include <omp.h> // OpenMP include file -0

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
    int ii, jj;  // Declare ii and jj here

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby; // add halo cells i think

    #pragma omp parallel for shared(u, utmp) private(ii, jj, diff) reduction(+:sum) // OpenMP pragma for loop with reduction clause
    for (ii=0; ii<nbx; ii++) // run over the row blocks
    {
        for (jj=0; jj<nby; jj++) // run over the col blocks
        {
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
            {
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) 
                {
                    utmp[i*sizey+j]= 0.25 * (
                                            u[ i*sizey     + (j-1) ]+  // left
                                            u[ i*sizey     + (j+1) ]+  // right
                                            u[ (i-1)*sizey + j     ]+  // top
                                            u[ (i+1)*sizey + j     ]
                                            ); // bottom

                    diff = utmp[i*sizey+j] - u[i*sizey + j];
                    sum += diff * diff; 
                }
            }
        }
    }

    // Halo management
    #pragma omp parallel for shared(u, utmp) private(ii, jj) reduction(+:sum) // OpenMP pragma for loop with reduction clause
    for (ii = 0; ii < nbx; ii++) 
    {
        for (jj = 0; jj < nby; jj++) 
        {
            // Top halo
            if (ii == 0) 
            {
                for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) 
                {
                    utmp[j] = u[j];
                }
            }
            // Bottom halo
            if (ii == nbx - 1) 
            {
                for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) 
                {
                    utmp[(sizex - 1) * sizey + j] = u[(sizex - 1) * sizey + j];
                }
            }
            // Left halo
            if (jj == 0) 
            {
                for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++) 
                {
                    utmp[i * sizey] = u[i * sizey];
                }
            }
            // Right halo
            if (jj == nby - 1) 
            {
                for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++) 
                {
                    utmp[i * sizey + sizey - 1] = u[i * sizey + sizey - 1];
                }
            }
        }
    }

    // Update u with utmp (required because we made changes to utmp)
    #pragma omp parallel for shared(u, utmp) private(ii, jj)
    for (ii=0; ii<nbx; ii++) // run over the row blocks
    {
        for (jj=0; jj<nby; jj++) // run over the col blocks
        {
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
            {
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) 
                {
                    u[i*sizey+j] = utmp[i*sizey+j];
                }
            }
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
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
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

    return sum;
}

