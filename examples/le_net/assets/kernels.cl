inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
   union{
       unsigned int u32;
       float        f32;
   } next, expected, current;
current.f32    = *addr;
   do{
   expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
	current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                           expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}


__kernel void convolution( const int firstRows,
                            const int firstCols,
                            const int secondRows,
                            const int secondCols,
                            const int numFilters,
                            const int sizeFilters,
                            const __global float* in,
                            const __global float* filters,
                            __global float* outs)
{
    const int globalFil = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(2);
    
    float acc = 0;
    int offsetOut = globalFil * secondRows * secondCols;
    int offsetFil = globalFil * sizeFilters * sizeFilters;
    
    for (int r = 0; r < sizeFilters;  r++)
    {
        for (int k = 0; k < sizeFilters; k++)
        {
            acc += in[(globalCol + k) * firstRows + globalRow + r] * filters[offsetFil + k * sizeFilters + r];
        }
    }
    outs[offsetOut + globalCol * secondRows + globalRow] = acc;
}


__kernel void back_convolution( const int firstRows,
                                const int firstCols,
                                const int secondRows,
                                const int secondCols,
                                const int numFilters,
                                const int sizeFilters,
                                const __global float* inA,
                                const __global float* inB,
                                __global float* outs)
{
    const int globalFil = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(2);
    
    float acc = 0;
    int offsetOut = globalFil * sizeFilters * sizeFilters;
    int offsetFil = globalFil * sizeFilters * sizeFilters;
    
    for (int r = 0; r < secondRows;  r++)
    {
        for (int k = 0; k < secondCols; k++)
        {
            acc += inA[(globalCol + k) * firstRows + globalRow + r] * inB[offsetFil + k * secondCols + r];
        }
    }
    outs[offsetOut + globalCol * sizeFilters + globalRow] = acc;
}


/* This is how it should be constructed, but for now just go by moduo
       0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
    0  X               X   X   X           X   X   X   X       X   X
    1  X   X               X   X   X           X   X   X   X       X
    2  X   X   X               X   X   X           X       X   X   X
    3      X   X   X           X   X   X   X           X       X   X
    4          X   X   X           X   X   X   X       X   X       X
    5              X   X   X           X   X   X   X       X   X   X
*/
__kernel void convolution16(const int firstRows,
                            const int firstCols,
                            const int secondRows,
                            const int secondCols,
                            const int sizeFilters,
                            const __global float* in,
                            const __global float* filters,
                            __global float* outs)
{
    const int globalFil = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(2);
    
    float acc = 0;
    int offsetIn = (globalFil % 6) * firstRows * firstCols;
    int offsetOut = globalFil * secondRows * secondCols;
    int offsetFil = globalFil * sizeFilters * sizeFilters;
    
    for (int r = 0; r < sizeFilters;  r++)
    {
        for (int k = 0; k < sizeFilters; k++)
        {
            acc += in[offsetIn + (globalCol + k) * firstRows + globalRow + r] * filters[offsetFil + k * sizeFilters + r];
        }
    }
    outs[offsetOut + globalCol * secondRows + globalRow] = acc;
}

__kernel void back_convolution16(const int firstRows,
                            const int firstCols,
                            const int secondRows,
                            const int secondCols,
                            const int sizeFilters,
                            const __global float* inA,
                            const __global float* inB,
                            __global float* outs)
{
    const int globalFil = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(2);
    
    int outOffset = globalFil * sizeFilters * sizeFilters;
    int inAOffset = (globalFil % 6) * firstRows * firstCols; // <-- let it stay this ways for now
    int inBOffset = globalFil * secondRows * secondCols;
    
    float acc = 0;
    
    for (int r = 0; r < secondRows;  r++)
    {
        for (int k = 0; k < secondCols; k++)
        {
            acc += inA[inAOffset + (globalCol + k) * firstRows + globalRow + r] * inB[inBOffset + k * secondCols + r];
        }
    }
    outs[outOffset + globalCol * sizeFilters + globalRow] = acc;
}


__kernel void deconvolution16(  const int firstRows,
                                const int firstCols,
                                const int secondRows,
                                const int secondCols,
                                const int sizeFilters,
                                const __global float* in,
                                const __global float* filters,
                                __global float* outs)
{
    const int globalFil = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(2);
    
    float acc = 0;
    int offsetIn =  globalFil * firstRows * firstRows;
    int offsetOut = (globalFil % 6) * secondRows * secondRows;
    int offsetFil = globalFil * sizeFilters * sizeFilters;
    
    for (int r = 0; r < sizeFilters;  r++)
    {
        for (int k = 0; k < sizeFilters; k++)
        {
             //atomicAdd_g_f(&outs[offsetOut + (globalCol + k) * secondRows + globalRow + r], 1);
             atomicAdd_g_f(&outs[offsetOut + (globalCol + k) * secondRows + globalRow + r], 
                in[offsetIn + globalCol * firstRows + globalRow] * filters[offsetFil + k * sizeFilters + r]);
        }
    }
}
                            


__kernel void matrix_multiply(  const int firstRows,
                                const int firstCols,
                                const int secondCols,
                                const __global float* inA,
                                const __global float* inB,
                                __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    float acc = 0.0f;
    
    for (int k = 0; k < firstCols; k++)
    {
        acc += inA[k * firstRows + globalRow] * inB[globalCol * firstCols + k];
    }
    
    out[globalCol * firstRows + globalRow] = acc;
}


__kernel void sigmoid(  const int rows,
                        const int cols,
                        __global float* in,
                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * rows + globalRow] = 1/(1+exp(-in[globalCol * rows + globalRow]));
}


__kernel void maxpool(  const int rows,
                        const int cols,
                        const __global float* in,
                        __global float* ind,
                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    int index = 0;
    float max = in[2*(globalCol * rows + globalRow)];
    
    if (in[2*(globalCol * rows + globalRow) + 1] > max)
    {
        max = in[2*(globalCol * rows + globalRow) + 1];
        index = 1;
    }
    else if (in[2*(globalCol * rows + globalRow) + rows] > max)
    {
        max = in[2*(globalCol * rows + globalRow) + rows];
        index = 2;
    }
    else if (in[2*(globalCol * rows + globalRow) + rows + 1] > max)
    {
        max = in[2*(globalCol * rows + globalRow) + rows + 1];
        index = 3;
    }

    ind[globalCol * rows + globalRow] = index;
    out[globalCol * rows + globalRow] = max;
}


__kernel void maxpool_error(const int rows,
                            const int cols,
                            const __global float* in,
                            const __global float* ind,
                            __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    int index = ind[globalRow * rows + globalCol];
     
    if (index == 0)
    {
        out[2 * (globalCol * 2 * rows + globalRow)] = in[globalRow * rows + globalCol];
        out[2 * (globalCol * 2 * rows + globalRow) + 1] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows) + 1] = 0;               
    }
    else if (index == 1)
    {
        out[2 * (globalCol * 2 * rows + globalRow)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow) + 1] = in[globalRow * rows + globalCol];
        out[2 * (globalCol * 2 * rows + globalRow + rows)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows) + 1] = 0;                             
    }
    else if (index == 2)
    {
        out[2 * (globalCol * 2 * rows + globalRow)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow) + 1] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows)] = in[globalRow * rows + globalCol];
        out[2 * (globalCol * 2 * rows + globalRow + rows) + 1] = 0;                            
    }
    else
    {
        out[2 * (globalCol * 2 * rows + globalRow)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow) + 1] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows)] = 0;
        out[2 * (globalCol * 2 * rows + globalRow + rows) + 1] = in[globalRow * rows + globalCol];                             
    }

}


__kernel void sigmoid_derivative(   const int rows,
                                    const int cols,
                                    const __global float* in,
                                    __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * rows + globalRow] = in[globalCol * rows + globalRow] * (1 - in[globalCol * rows + globalRow]);
}


__kernel void matrix_subtract(  const int rows,
                                const int cols,
                                const __global float* inA,
                                const __global float* inB,
                                __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * rows + globalRow] = inA[globalCol * rows + globalRow] - inB[globalCol * rows + globalRow];
}


__kernel void matrix_point_multiply(const int rows,
                                    const int cols,
                                    const __global float* inA,
                                    const __global float* inB,
                                    __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * rows + globalRow] = inA[globalCol * rows + globalRow] * inB[globalCol * rows + globalRow];
}


__kernel void matrix_transpose_multiply(const int firstRows,
                                        const int firstCols,
                                        const int secondCols,
                                        const __global float* inA,
                                        const __global float* inB,
                                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    float acc = 0.0f;
    
    for (int k = 0; k < firstRows; k++)
    {
        acc += inA[globalCol * firstRows + k] * inB[globalCol * firstRows + k];
    }
    
    out[globalCol * firstCols + globalRow] = acc;
}


__kernel void matrix_multiply_transpose(const int firstRows,
                                        const int firstCols,
                                        const int secondRows,
                                        const __global float* inA,
                                        const __global float* inB,
                                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    float acc = 0.0f;
    
    for (int k = 0; k < firstCols; k++)
    {
        acc += inA[k * firstRows + globalRow] * inB[globalCol * firstCols + k];
    }
    
    out[globalCol * firstRows + globalRow] = acc;
}


__kernel void matrix_add(   const int rows,
                            const int cols,
                            __global float* inA,
                            __global float* inB,
                            __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * rows + globalRow] = inA[globalCol * rows + globalRow] + inB[globalCol * rows + globalRow];
}
