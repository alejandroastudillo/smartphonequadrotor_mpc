#include <jni.h>
#include <string>

#include "mpc_c.h"
#include <time.h>
#include <stdio.h>
#include <exception>

#include <sstream>

extern "C" JNIEXPORT jstring JNICALL
Java_com_calileu_quadrotor_1nmpc_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    mpc_step_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];
    casadi_int n_in = mpc_step_n_in();
    casadi_int n_out = mpc_step_n_out();
    const casadi_real* arg[sz_arg];
    casadi_real* res[sz_res];


    // Get total number of input nonzeros
    casadi_int nnz_in = 0;
    for (casadi_int i=0;i<n_in;++i) {
        const casadi_int* sp_in = mpc_step_sparsity_in(i);
        nnz_in += sp_in[2+sp_in[1]];
    }

    // Get total number of output nonzeros
    casadi_int nnz_out = 0;
    for (casadi_int i=0;i<n_out;++i) {
        const casadi_int* sp_out = mpc_step_sparsity_out(i);
        nnz_out += sp_out[2+sp_out[1]];
    }

    // Read stdin to input buffer
    casadi_real w_in[nnz_in];
    casadi_real* a = w_in;
    for (casadi_int j=0; j<nnz_in; ++j) scanf("%lg", a++);

    // Point arguments vector entries to location of input nonzeros
    casadi_int offset = 0;
    for (casadi_int i=0;i<n_in;++i) {
        arg[i]=w_in+offset;
        const casadi_int* sp_in = mpc_step_sparsity_in(i);
        casadi_int nnz_in = sp_in[2+sp_in[1]];
        printf("nnz %lld: %lld\n",i, nnz_in);
        offset+= nnz_in;
    }

    // Point result vector entries to location of input nonzeros
    casadi_real w_out[nnz_out];
    offset = 0;
    for (casadi_int i=0;i<n_in;++i) {
        res[i]=w_out+offset;
        const casadi_int* sp_out = mpc_step_sparsity_out(i);
        casadi_int nnz_out = sp_out[2+sp_out[1]];
        offset+= nnz_out;
    }

    clock_t start_t, end_t;

    int N = 100;

    start_t = clock();
    //printf("Starting of the program, start_t = %ld\n", start_t);

    casadi_int flag;
    for(int j= 1; j < N; j++) {
        flag = mpc_step(arg, res, iw, w, 0);
    }
    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);

    const casadi_real* b = w_out;
    for (casadi_int j=0; j<nnz_out; ++j) printf("%g ", *b++);
    printf("\n");

    double total_t = (end_t - start_t) / (double) CLOCKS_PER_SEC;
    double avgCPUtime = total_t/N*1e6;
    printf("Total time taken by CPU: %f s\n", total_t);
    printf("Average CPU time: %f us (over %d trials)\n", total_t/N*1e6, N);

    std::ostringstream strs;
    strs << "Total time taken by CPU: " << total_t << "\n\n Average CPU time: " << avgCPUtime << " us (over " << N << " trials)";
    std::string str = strs.str();

    return env->NewStringUTF(str.c_str());
}
