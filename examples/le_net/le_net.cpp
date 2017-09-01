#include "common.h"
#include "image.h"

#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <cstring>

using namespace std;
using namespace chrono;

/*
Memory objects needed:

        Layer1          Layer 2         Layer 3         Layer 4         Layer 5         Layer 6         Layer 7
    Infere______________________________________________________________________________________________________________
        - image 32x32   - y 6@14x14     - syn 16@5x5    - y 16@5x5      - syn 400x120   - syn 120x84    - syn 84x10
        - syn 6@5x5     - a 6@14x14     - y 16@10x10    - a 16@5x5      - y 120         - y 84          - y 10
        - y 6@28x28                     - a 16@10x10                    - a 120         - a 84          - a 10
        - a 6@28x28
    Train________________________________________________________________________________________________________________
        - e 6@28x28     - e 6@14x14     - e 16@10x10    - e 16@5x5      - e 120         - e 84          - output 10
        - g 6@5x5                       - g 16@10x10                    - g 120         - g 84          - e 10
        - d 6@5x5                       - d 16@10x10                    - d 120         - d 84          - g 10
        - dsyn 6@5x5                    - dsyn 16@5x5                   - dsyn 400x120  - dsyn 120x84   - d 10
                                                                                                        - dsyn 84x10 

Operations needed:
    Infere:
        - L1_y = convolution6(image, L1_syn)  <-- Convolution with 6 different filters
        - L1_a = sigmoid(L1_y)                <-- 1/1+exp(-L1_y)
        
          
        - L2_a, L2_y = maxpool(L1_a)          <-- Pools max value from 4 pixels and remember which pixel from four inputs is max
        
        - L3_y = convolution16(L2_a, L3_syn)  <-- Convolution with 16 different filters that are strangly connected
        - L3_a = sigmoid(L3_y)
        
        - L4_a, L4_y = maxpool(L3_a)
        
        - L5_y = multiply(L4_a, L5_syn)
        - L5_a = sigmoid(L5_y)
        
        - L6_y = multiply(L5_a, L6_syn)
        - L6_a = sigmoid(L6_y)
        
        - L7_y = multiply(L6_a, L7_syn)
        - L7_a = sigmoid(L7_y)
    
    Train:
        - L7_e = subtract(output, L7_a)           <-- Subtract elements of first matrix from second
        - L7_g = sigmoid_gradient(L7_a)           <-- Find sigmoid gradient for every element
        - L7_d = pointwise_multiply(L7_g, L7_e)       <-- Multiply all coresponding elements
        - L7_dsyn = transpose_multiply(L6_a, L7_d)    <-- Transpose first matrix and multiply it with second
        
        - L6_e = multiply_transpose(L7_d, L7_syn)     <-- Transpose second matrix and multiply first with second
        - L6_g = sigmoid_gradient(L6_a)
        - L6_d = pointwise_multiply(L6_g, L6_e)
        - L6_dsyn = transpose_multiply(L5_a, L6_d)

        - L5_e = multiply_transpose(L6_d, L6_syn)
        - L5_g = sigmoid_gradient(L5_a)
        - L5_d = pointwise_multiply(L5_g, L5_e)
        - L5_dsyn = transpose_multiply(L4_a, L5_d)
        
        - L4_e = multiply_transpose(L5_d, L5_syn)
        
        - L3_e = maxpool_error(L4_e, L4_y)
        - L3_g = sigmoid_gradient(L3_a)
        - L3_d = pointwise_multiply(L3_e, L3_g)
        - L3_dsyn = back_convolution16(L2_a, L3_d)
        
        - L2_e = deconvolution(L3_d, L3_syn)
        
        - L1_e = maxpool_error(L2_e, L2_y)
        - L1_g = sigmoid_gradient(L1_a)
        - L1_d = pointwise_multiply(L1_e, L1_g)
        L1_dsyn = back_convolution6(L1_d, image)
*/

#define TEST_IND 41
#define NUMBER_OF_OPERATIONS 14
#define SIZE 10

int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernels[NUMBER_OF_OPERATIONS] = {0};
    cl_event event = 0;
    
    int numberOfMemoryObjects = 45;
    cl_mem memoryObjects[45] = {0};
    cl_int errorNumber;
    string kernel_names[] = {"convolution", "sigmoid", "maxpool", "convolution16", "matrix_multiply", "matrix_subtract", 
        "sigmoid_derivative", "matrix_point_multiply", "matrix_transpose_multiply", "matrix_multiply_transpose", 
        "maxpool_error", "back_convolution16", "deconvolution16", "back_convolution"};
    
    size_t firstRows, firstCols, secondRows, secondCols, numFilters, filterSize;   
    size_t globalWorksize2[2];
    size_t globalWorksize3[3];
    
    const size_t localWorksize2[2] = {1, 1};
    const size_t localWorksize3[3] = {1, 1, 1};
    bool setKernelArgumentsSuccess = true;
    
    /*  Prepare context, command queue, program and kernels
        NOTE: We wont use most of clean functions, as the code will be just huge. */
        
    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createProgram(context, device, "assets/kernels.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    for (int i = 0; i < NUMBER_OF_OPERATIONS; i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &errorNumber);
        
        if (!checkSuccess(errorNumber))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }
    }
    
    /* Ask the OpenCL implementation to allocate buffers for the data */ 
    bool createMemoryObjectsSuccess = true;  
    size_t buffSizes[] = {1024, 150, 4704, 4704, 1176, 1176, 400, 1600, 1600, 400,
                          400, 48000, 120, 120, 10080, 84, 84, 840, 10, 10,
                          10, 10, 10, 10, 840, 84, 84, 84, 1080, 120,
                          120, 120, 48000, 400, 1600, 1600, 1600, 400, 1176, 4704,
                          4704, 4704, 150};
    
    for (int i = 0; i < numberOfMemoryObjects; i++)
    {    
        memoryObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffSizes[i] * sizeof(float), NULL, &errorNumber);
        createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    }   
    
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffer. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU */
    bool mapMemoryObjectsSuccess = true;
    
    cl_float* image = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[0], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* L1_syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[1], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    cl_float* L3_syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[6], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[6], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    cl_float* L5_syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[11], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[11], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* L6_syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[14], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[14], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    cl_float* L7_syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[17], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[17], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    cl_float* output = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[20], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[20], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
        
    if (mapMemoryObjectsSuccess == false)
        cout << "ERROR!" << endl;
            
    /* Initialize the data */        
    for (unsigned int i=0; i<buffSizes[0]; i++)
        image[i] = 1;
        
    for (unsigned int i=0; i<buffSizes[1]; i++)
        L1_syn[i] = 0.01;

    for (unsigned int i=0; i<buffSizes[6]; i++)
        L3_syn[i] = 0.01;

    for (unsigned int i=0; i<buffSizes[11]; i++)
        L5_syn[i] = 0.01;

    for (unsigned int i=0; i<buffSizes[14]; i++)
        L6_syn[i] = 0.01;

    for (unsigned int i=0; i<buffSizes[17]; i++)
        L7_syn[i] = 0.01;

    for (unsigned int i=0; i<buffSizes[20]; i++)
        output[i] = 3;
                 
    /* Unmap buffers, so GPU can use them */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], image, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], L1_syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[6], L3_syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
        
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[11], L5_syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
        
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[14], L6_syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
        
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[17], L7_syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
        
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[20], output, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    } 
       
    /* L1_y = convolution6(image, L1_syn)  <-- Convolution with 6 different filters */
    firstRows = 32;
    firstCols = 32;
    secondRows = 28;
    secondCols = 28;
    numFilters = 6;
    filterSize = 5;
     
    globalWorksize3[0] = numFilters;
    globalWorksize3[1] = secondRows;
    globalWorksize3[2] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 3, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 4, sizeof(int), (void*)&numFilters));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 5, sizeof(int), (void*)&filterSize));       
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 6, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 7, sizeof(cl_mem), (void*)&memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 8, sizeof(cl_mem), (void*)&memoryObjects[2]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[0], 3, NULL, globalWorksize3, localWorksize3, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L1_a = sigmoid(L1_y)                <-- 1/1+exp(-L1_y) */
    firstRows = 28;
    firstCols = 168;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[3]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L2_a, L2_y = maxpool(L1_a)            <-- Pools max value from 4 pixels and remember which pixel from four inputs is max */
    firstRows = 14;
    firstCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), (void*)&memoryObjects[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 3, sizeof(cl_mem), (void*)&memoryObjects[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 4, sizeof(cl_mem), (void*)&memoryObjects[5]));    
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[2], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }    

    /* L3_y = convolution16(L2_a, L3_syn) */
    firstRows = 14;
    firstCols = 14;
    secondRows = 10;
    secondCols = 10;
    numFilters = 16;
    filterSize = 5;
     
    globalWorksize3[0] = numFilters;
    globalWorksize3[1] = secondRows;
    globalWorksize3[2] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 3, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 4, sizeof(int), (void*)&filterSize));       
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 5, sizeof(cl_mem), (void*)&memoryObjects[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 6, sizeof(cl_mem), (void*)&memoryObjects[6]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 7, sizeof(cl_mem), (void*)&memoryObjects[7]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[3], 3, NULL, globalWorksize3, localWorksize3, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L3_a = sigmoid(L3_y) */
    firstRows = 10;
    firstCols = 160;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[7]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[8]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L4_a, L4_y = maxpool(L1_a) */
    firstRows = 5;
    firstCols = 80;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), (void*)&memoryObjects[8]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 3, sizeof(cl_mem), (void*)&memoryObjects[9]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 4, sizeof(cl_mem), (void*)&memoryObjects[10]));    
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[2], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L5_y = multiply(L4_a, L5_syn) */
    firstRows = 1;
    firstCols = 400;
    secondCols = 120;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 3, sizeof(cl_mem), (void*)&memoryObjects[10]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 4, sizeof(cl_mem), (void*)&memoryObjects[11]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 5, sizeof(cl_mem), (void*)&memoryObjects[12]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[4], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L5_a = sigmoid(L5_y) */
    firstRows = 1;
    firstCols = 120;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[12]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[13]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    } 
    
    /* L6_y = multiply(L5_a, L6_syn) */
    firstRows = 1;
    firstCols = 120;
    secondCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 3, sizeof(cl_mem), (void*)&memoryObjects[13]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 4, sizeof(cl_mem), (void*)&memoryObjects[14]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 5, sizeof(cl_mem), (void*)&memoryObjects[15]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[4], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L6_a = sigmoid(L6_y) */
    firstRows = 1;
    firstCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[15]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[16]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    } 
    
    /* L7_y = multiply(L6_a, L7_syn) */
    firstRows = 1;
    firstCols = 84;
    secondCols = 10;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 3, sizeof(cl_mem), (void*)&memoryObjects[16]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 4, sizeof(cl_mem), (void*)&memoryObjects[17]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 5, sizeof(cl_mem), (void*)&memoryObjects[18]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[4], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L7_a = sigmoid(L7_y) */
    firstRows = 1;
    firstCols = 10;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[18]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[19]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L7_e = subtract(output, L7_a) */
    firstRows = 1;
    firstCols = 10;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 2, sizeof(cl_mem), (void*)&memoryObjects[20]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 3, sizeof(cl_mem), (void*)&memoryObjects[19]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 4, sizeof(cl_mem), (void*)&memoryObjects[21]));
           
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[5], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[5], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[5], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L7_g = sigmoid_derivative(L7_a) */
    firstRows = 1;
    firstCols = 10;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), (void*)&memoryObjects[19]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[22]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L7_d = pointwise(L7_e, L7_g) */
    firstRows = 1;
    firstCols = 10;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[21]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[22]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[23]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L7_dsyn = transpose_multiply(L6_a, L7_syn) */
    firstRows = 1;
    firstCols = 84;
    secondCols = 10;
     
    globalWorksize2[0] = firstCols;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 3, sizeof(cl_mem), (void*)&memoryObjects[19]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 4, sizeof(cl_mem), (void*)&memoryObjects[23]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 5, sizeof(cl_mem), (void*)&memoryObjects[24]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[8], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L6_e = multiply_transpose(L7_d, L7_syn) */
    firstRows = 1;
    firstCols = 10;
    secondRows = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondRows;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 3, sizeof(cl_mem), (void*)&memoryObjects[23]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 4, sizeof(cl_mem), (void*)&memoryObjects[17]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 5, sizeof(cl_mem), (void*)&memoryObjects[25]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[9], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L6_g = sigmoid_derivative(L6_a) */
    firstRows = 1;
    firstCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), (void*)&memoryObjects[16]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[26]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L6_d = pointwise(L6_e, L6_g) */
    firstRows = 1;
    firstCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[25]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[26]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[27]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L6_dsyn = transpose_multiply(L5_a, L6_syn) */
    firstRows = 1;
    firstCols = 120;
    secondCols = 84;
     
    globalWorksize2[0] = firstCols;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 3, sizeof(cl_mem), (void*)&memoryObjects[13]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 4, sizeof(cl_mem), (void*)&memoryObjects[27]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 5, sizeof(cl_mem), (void*)&memoryObjects[28]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[8], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L5_e = multiply_transpose(L6_d, L6_syn) */
    firstRows = 1;
    firstCols = 84;
    secondRows = 120;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondRows;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 3, sizeof(cl_mem), (void*)&memoryObjects[27]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 4, sizeof(cl_mem), (void*)&memoryObjects[14]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 5, sizeof(cl_mem), (void*)&memoryObjects[29]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[9], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L5_g = sigmoid_derivative(L5_a) */
    firstRows = 1;
    firstCols = 120;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), (void*)&memoryObjects[13]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[30]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* L5_d = pointwise(L5_e, L5_g) */
    firstRows = 1;
    firstCols = 120;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[29]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[30]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[31]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L5_dsyn = transpose_multiply(L4_a, L5_syn) */
    firstRows = 1;
    firstCols = 400;
    secondCols = 120;
     
    globalWorksize2[0] = firstCols;
    globalWorksize2[1] = secondCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 2, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 3, sizeof(cl_mem), (void*)&memoryObjects[10]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 4, sizeof(cl_mem), (void*)&memoryObjects[31]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[8], 5, sizeof(cl_mem), (void*)&memoryObjects[32]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[8], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[8], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L4_e = multiply_transpose(L5_d, L5_syn) */
    firstRows = 1;
    firstCols = 120;
    secondRows = 400;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = secondRows;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 3, sizeof(cl_mem), (void*)&memoryObjects[31]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 4, sizeof(cl_mem), (void*)&memoryObjects[11]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[9], 5, sizeof(cl_mem), (void*)&memoryObjects[33]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[9], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[9], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }    

    /* L3_e = maxpool_error(L4_e, L4_y) */
    firstRows = 5;
    firstCols = 80;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 2, sizeof(cl_mem), (void*)&memoryObjects[33]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 3, sizeof(cl_mem), (void*)&memoryObjects[9]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 4, sizeof(cl_mem), (void*)&memoryObjects[34]));    
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[10], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[10], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[10], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L3_g = sigmoid_derivative(L3_a) */
    firstRows = 10;
    firstCols = 160;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), (void*)&memoryObjects[8]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[35]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L3_d = pointwise(L3_e, L3_g) */
    firstRows = 10;
    firstCols = 160;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[34]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[35]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[36]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L3_dsyn = deconvolution16(L2_a, L3_d) */
    firstRows = 14;
    firstCols = 14;
    secondRows = 10;
    secondCols = 10;
    numFilters = 16;
    filterSize = 5;
     
    globalWorksize3[0] = numFilters;
    globalWorksize3[1] = filterSize;
    globalWorksize3[2] = filterSize;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 3, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 4, sizeof(int), (void*)&filterSize));       
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 5, sizeof(cl_mem), (void*)&memoryObjects[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 6, sizeof(cl_mem), (void*)&memoryObjects[36]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[11], 7, sizeof(cl_mem), (void*)&memoryObjects[37]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[11], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[11], 3, NULL, globalWorksize3, localWorksize3, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[11], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L2_e = deconvolution16(L3_d, L3_syn) */
    firstRows = 10;
    firstCols = 1600;
    secondRows = 14;
    secondCols = 84;
    numFilters = 16;
     
    globalWorksize3[0] = 16;
    globalWorksize3[1] = 10;
    globalWorksize3[2] = 10;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 3, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 4, sizeof(int), (void*)&filterSize));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 5, sizeof(cl_mem), (void*)&memoryObjects[36]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 6, sizeof(cl_mem), (void*)&memoryObjects[6]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[12], 7, sizeof(cl_mem), (void*)&memoryObjects[38]));        
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[12], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[12], 3, NULL, globalWorksize3, localWorksize3, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[12], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L1_e = maxpool_error(L2_e, L2_y) */
    firstRows = 14;
    firstCols = 84;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 2, sizeof(cl_mem), (void*)&memoryObjects[38]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 3, sizeof(cl_mem), (void*)&memoryObjects[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[10], 4, sizeof(cl_mem), (void*)&memoryObjects[39]));    
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[10], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[10], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[10], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L1_g = sigmoid_derivative(L1_a) */
    firstRows = 28;
    firstCols = 168;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), (void*)&memoryObjects[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[40]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L1_d = pointwise(L1_e, L1_g) */
    firstRows = 28;
    firstCols = 168;
     
    globalWorksize2[0] = firstRows;
    globalWorksize2[1] = firstCols;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&firstCols));      
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[39]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[40]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[41]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize2, localWorksize2, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* L1_dsyn = back_convolution(image, L1_d) */
    firstRows = 32;
    firstCols = 32;
    secondRows = 28;
    secondCols = 28;
    numFilters = 6;
    filterSize = 5;
     
    globalWorksize3[0] = numFilters;
    globalWorksize3[1] = filterSize;
    globalWorksize3[2] = filterSize;
    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 0, sizeof(int), (void*)&firstRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 1, sizeof(int), (void*)&firstCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 2, sizeof(int), (void*)&secondRows));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 3, sizeof(int), (void*)&secondCols));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 4, sizeof(int), (void*)&numFilters));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 5, sizeof(int), (void*)&filterSize));       
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 6, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 7, sizeof(cl_mem), (void*)&memoryObjects[41]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[13], 8, sizeof(cl_mem), (void*)&memoryObjects[42]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[13], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[13], 3, NULL, globalWorksize3, localWorksize3, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[13], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for command queue to finish, wait for event and release it */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    clWaitForEvents(1, &event);  
    
    if (!checkSuccess(clReleaseEvent(event)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    /* Map buffer to read results */        
    cl_float* res = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[TEST_IND], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[TEST_IND], 0, NULL, NULL, &errorNumber);
    
    if (!checkSuccess(errorNumber))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    cout << endl << "res: ";
    for (unsigned int i = 0; i < buffSizes[TEST_IND]; i++)
    {
        if (i%SIZE == 0)
            cout << endl << i/SIZE << ".\t";
            
        cout << res[i] << "\t";
    }
    cout << endl;

    /* Unmap memory objects */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[TEST_IND], res, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
}
