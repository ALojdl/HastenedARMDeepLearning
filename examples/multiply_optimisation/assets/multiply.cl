#define TS 8
#define WPT 4
// --------------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------------
__kernel void matrix_multiply_tiling(   const int M, 
                                        const int N,
                                        const int K,
                                        const __global float* inA,
                                        const __global float* inB,
                                        __global float* out)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    float acc = 0.0f;    
    const int numTiles = K / TS;

    for (int t = 0; t < numTiles; t++)
    {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        
        Asub[col][row] = inA[tiledCol * M + globalRow];
        Bsub[col][row] = inB[globalCol * K + tiledRow];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TS; k++)
        {
            acc += Asub[k][row] * Bsub[col][k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);     
    }
        
    out[globalCol * M + globalRow] = acc;
}
// --------------------------------------------------------------------------------------------
__kernel void matrix_multiply_less_loads(   const int M,
                                            const int N,
                                            const int K,
                                            const __global float* inA,
                                            const __global float* inB,
                                            __global float* out)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    const int numTiles = K/TS;        
    float acc[WPT];
    
    for (int t = 0; t < WPT; t++)
    {
        acc[t] = 0.0f;
    } 

    for (int t = 0; t < numTiles; t++)
    {
        for (int r = 0; r < WPT; r++)
        {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            
            Asub[col + r * WPT][row] = inA[(tiledCol + r * WPT) * M + globalRow];
            Bsub[col + r * WPT][row] = inB[(globalCol + r * WPT) * K + tiledRow];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TS; k++)
        {
            for (int r = 0; r < WPT; r++)
            {
                acc[r] += Asub[k][row] * Bsub[col + r * WPT][k];
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);     
    }
    
    for (int r = 0; r < WPT; r++)
    {
        out[(globalCol + r * WPT) * M + globalRow] = acc[r];
    }
}
// --------------------------------------------------------------------------------------------
// NOTE: Still work in progress!!! We should transpose second input matrix.
__kernel void matrix_multiply_vector(   const int M,
                                        const int N,
                                        const int K,
                                        const __global float* inA,
                                        const __global float* inB,
                                        __global float* out)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    __local float4 Asub[TS][TS/4];
    __local float4 Bsub[TS][TS/4];

    const int numTiles = K/TS;       
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < numTiles; t++)
    {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        
        Asub[col][row] = vload4((tiledCol * M + globalRow), inA);
        Bsub[col][row] = vload4((globalCol * K + tiledRow), inB);
    
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TS; k++)
        {
            acc += Asub[k][row] * Bsub[col][k]; 
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);     
    }
    
    vstore4(acc, (globalCol * M + globalRow), out);
}

