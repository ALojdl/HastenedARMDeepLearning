#include "common.h"
#include "image.h"

#include <CL/cl.h>
#include <iostream>

#define SIZE 8
#define LOC_W_SIZE 2

using namespace std;

int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernelAB = 0, kernelBC = 0, kernelCA = 0;
    cl_event event = 0;
    
    int numberOfMemoryObjects = 3;
    cl_mem memoryObjects[3] = {0, 0, 0};
    cl_int errorNumber;
    
    size_t M, N, K;
    M = N = K = SIZE;
    
    size_t globalWorksize[2] = {M, N};
    const size_t localWorksize[2] = {LOC_W_SIZE, LOC_W_SIZE};
    cl_int arraySize = SIZE * SIZE;
    size_t bufferSize = arraySize * sizeof(cl_float);
    bool setKernelArgumentsSuccess = true;

    /* Prepare context, command queue, program and kernels */
    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createProgram(context, device, "assets/multiply.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    kernelAB = clCreateKernel(program, "matrix_multiply", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    kernelBC = clCreateKernel(program, "matrix_multiply", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelBC, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    kernelCA = clCreateKernel(program, "matrix_multiply", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelCA, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Ask the OpenCL implementation to allocate buffers for the data */     
    bool createMemoryObjectsSuccess = true;
    
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    memoryObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffer. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU */
    bool mapMemoryObjectsSuccess = true;
    
    cl_float* A = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], 
        CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* B = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], 
        CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    
    cl_float* C = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], 
        CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    /* Initialize the data */
    for (int i = 0; i < arraySize; i++)
    {
       A[i] = 1.0f;
       B[i] = 1.0f;
       C[i] = 1.0f;
    }
    
    /* Unmap buffers, so GPU can use them */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], A, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], B, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], C, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    /* Set the kernel arguments for first matrix multiply and enqueue the kernel */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 1, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 2, sizeof(int), (void*)&K));    
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 3, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 4, sizeof(cl_mem), (void*)&memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelAB, 5, sizeof(cl_mem), (void*)&memoryObjects[2]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernelAB, 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Set the kernel arguments for second matrix multiply and enqueue the kernel */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 1, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 2, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 3, sizeof(cl_mem), (void*)&memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 4, sizeof(cl_mem), (void*)&memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelBC, 5, sizeof(cl_mem), (void*)&memoryObjects[0]));
    
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernelBC, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernelBC, 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelBC, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Set the kernel arguments for third matrix multiply and enqueue the kernel */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 1, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 2, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 3, sizeof(cl_mem), (void*)&memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 4, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernelCA, 5, sizeof(cl_mem), (void*)&memoryObjects[1]));
    
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernelCA, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernelCA, 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelBC, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for command queue to finish, wait for event and release it */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    clWaitForEvents(1 , &event);    
    
    if (!checkSuccess(clReleaseEvent(event)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    /* Map buffers again to read results, later enqueue again */
    A = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    B = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    C = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    
    if (!checkSuccess(errorNumber))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cout << "A: " << endl;
    for (int i = 0; i < arraySize; i++)
    {
        cout << "i = " << i << ", output = " <<  A[i] << endl;
    }

    cout << "B: " << endl;
    for (int i = 0; i < arraySize; i++)
    {
        cout << "i = " << i << ", output = " <<  B[i] << endl;
    }

    cout << "C: " << endl;
    for (int i = 0; i < arraySize; i++)
    {
        cout << "i = " << i << ", output = " <<  C[i] << endl;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], A, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], B, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], C, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cleanUpOpenCL(context, commandQueue, program, kernelAB, memoryObjects, numberOfMemoryObjects);
}
