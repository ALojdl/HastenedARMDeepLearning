__kernel void matrix_multiply(  const int M,
                                const int N,
                                const int K,
                                const __global float* inA,
                                const __global float* inB,
                                __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    float acc = 0.0f;
    
    for (int k = 0; k < K; k++)
    {
        acc += inA[k * M + globalRow] * inB[globalCol * K + k];
    }
    
    out[globalCol * M + globalRow] = acc;
}


__kernel void matrix_nonlin(const int M,
                            const int N,
                            __global float* in,
                            __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * M + globalRow] = 1/(1+exp(-in[globalCol * M + globalRow]));
}


__kernel void matrix_nonlin_derivative( const int M,
                                        const int N,
                                        __global float* in,
                                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * M + globalRow] = in[globalCol * M + globalRow] * (1 - in[globalCol * M + globalRow]);
}


__kernel void matrix_subtract(  const int M,
                                const int N,
                                __global float* inA,
                                __global float* inB,
                                __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * M + globalRow] = inA[globalCol * M + globalRow] - inB[globalCol * M + globalRow];
}


__kernel void matrix_point_multiply(    const int M,
                                        const int N,
                                        __global float* inA,
                                        __global float* inB,
                                        __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * M + globalRow] = inA[globalCol * M + globalRow] * inB[globalCol * M + globalRow];
}


__kernel void matrix_transpose( const int M,
                                const int N,
                                __global float* in,
                                __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalRow * N + globalCol] = in[globalCol * M + globalRow];
}


__kernel void matrix_add(   const int M,
                            const int N,
                            __global float* inA,
                            __global float* inB,
                            __global float* out)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    out[globalCol * M + globalRow] = inA[globalCol * M + globalRow] + inB[globalCol * M + globalRow];
}
