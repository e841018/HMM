#include "hmm.h"

int main(int argc, char *argv[]){
	
	// check number of arguments
	if(argc!=5){
		fprintf(stderr, "argc = %d\n", argc);
		return EXIT_FAILURE;
	}
	
	// input
	int n_iter = atoi(argv[1]);
	HMM hmm;
	loadHMM(&hmm, argv[2]);
	FILE *seq = open_or_die(argv[3], "r");
	FILE *output = open_or_die(argv[4], "w");
	
	// variables
	int N = hmm.state_num;
	int O = hmm.observ_num;
	int T = 0;			// length of each line read from seq
	int n_sample = 0;	// number of lines in seq
	int obs[MAX_SEQ];	// buffer for a line from seq
	
	// dynamically allocated variables
	double **alpha = malloc_2d(N, MAX_SEQ);
	double **beta = malloc_2d(N, MAX_SEQ);
	double **gamma = malloc_2d(N, MAX_SEQ);
	double ***epsilon = malloc_3d(MAX_SEQ-1, N, N);
	double *gamma1_sum = malloc(sizeof(double)*N);
	double *gamma_sum = malloc(sizeof(double)*N);
	double **epsilon_sum = malloc_2d(N,N);
	double *gamma_all_sum = malloc(sizeof(double)*N);
	double **gamma_obs_sum = malloc_2d(O,N);
	
	// train
	for(int iter=0; iter<n_iter; iter++){
		
		// clear sums
		// memset(gamma1_sum, 0, sizeof(double)*N);
		// memset(gamma_sum, 0, sizeof(double)*N);
		// memset(epsilon_sum+N, 0, sizeof(double)*N*N);
		// memset(gamma_all_sum, 0, sizeof(double)*N);
		// memset(gamma_obs_sum+O, 0, sizeof(double)*O*N);
		for(int i=0; i<N; i++)
			gamma1_sum[i] = 0;
		for(int i=0; i<N; i++)
			gamma_sum[i] = 0;
		for(int i=0; i<N; i++)
			for(int j=0; j<N; j++)
				epsilon_sum[i][j] = 0;
		for(int i=0; i<N; i++)
			gamma_all_sum[i] = 0;
		for(int o=0; o<O; o++)
			for(int i=0; i<N; i++)
				gamma_obs_sum[o][i] = 0;
		n_sample = 0;
			
		// loop through each line in training data
		rewind(seq);
		while((T = get_obs(seq, obs))!=-1){
			n_sample++;
			
			// alpha
			for(int i=0; i<N; i++)
				alpha[i][0] = hmm.initial[i] * hmm.observation[obs[0]][i];
			for(int t=1; t<T; t++)
				for(int i=0; i<N; i++){
					double sum = 0;
					for(int j=0; j<N; j++)
						sum += alpha[j][t-1] * hmm.transition[j][i];
					alpha[i][t] = sum * hmm.observation[obs[t]][i];
			}
			
			// beta
			for(int i=0; i<N; i++)
				beta[i][T-1] = 1;
			for(int t=T-2; t>=0; t--)
				for(int i=0; i<N; i++){
					double sum = 0;
					for(int j=0; j<N; j++)
						sum += hmm.transition[i][j]
							* hmm.observation[obs[t+1]][j] * beta[j][t+1];
					beta[i][t] = sum;
			}
			
			// gamma
			for(int t=0; t<T; t++){
				double sum = 0;
				for(int i=0; i<N; i++){
					gamma[i][t] = alpha[i][t] * beta[i][t];
					sum += gamma[i][t];
				}
				for(int i=0; i<N; i++)
					gamma[i][t] /= sum;
			}
			
			// epsilon
			for(int t=0; t<T-1; t++){
				double sum = 0;
				for(int i=0; i<N; i++)
					for(int j=0; j<N; j++){
						epsilon[t][i][j] = alpha[i][t] * hmm.transition[i][j]
							* hmm.observation[obs[t+1]][j] * beta[j][t+1];
						sum += epsilon[t][i][j];
				}
				for(int i=0; i<N; i++)
					for(int j=0; j<N; j++)
						epsilon[t][i][j] /= sum;
			}
			
			// accumulate sums
			for(int i=0; i<N; i++)
				gamma1_sum[i] += gamma[i][0];
			for(int i=0; i<N; i++)
				for(int t=0; t<T-1; t++)
					gamma_sum[i] += gamma[i][t];
			for(int t=0; t<T-1; t++)
				for(int i=0; i<N; i++)
					for(int j=0; j<N; j++)
						epsilon_sum[i][j] += epsilon[t][i][j];
			for(int i=0; i<N; i++)
				for(int t=0; t<T; t++){
					gamma_all_sum[i] += gamma[i][t];
					gamma_obs_sum[obs[t]][i] += gamma[i][t];
			}
		}
		
		// re-estimate model
		for(int i=0; i<N; i++)
			hmm.initial[i] = gamma1_sum[i]/n_sample;
		for(int i=0; i<N; i++)
			for(int j=0; j<N; j++)
				hmm.transition[i][j] = epsilon_sum[i][j]/gamma_sum[i];
		for(int o=0;o<O; o++)
			for(int i=0; i<N; i++)
				hmm.observation[o][i] = gamma_obs_sum[o][i]/gamma_all_sum[i];
	}
	
	// store trained model
	dumpHMM(output, &hmm);
	
	// clean up
	free(gamma_obs_sum);
	free(gamma_all_sum);
	free(epsilon_sum);
	free(gamma_sum);
	free(gamma1_sum);
	free_3d(epsilon, MAX_SEQ-1);
	free(gamma);
	free(beta);
	free(alpha);
	fclose(seq);
	fclose(output);
	return 0;
}