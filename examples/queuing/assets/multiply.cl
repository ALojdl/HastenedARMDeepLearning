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
