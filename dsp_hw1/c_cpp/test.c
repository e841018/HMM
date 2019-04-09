#include "hmm.h"

int main(int argc, char *argv[]){
	
	// check number of arguments
	if(argc!=4 && argc!=5){
		fprintf(stderr, "argc = %d\n", argc);
		return EXIT_FAILURE;
	}
	
	// calc accuracy on training set to check if overfitting occured
	int test_training_set = 0;
	if(argc==5)
		test_training_set = 1;
	
	// input
	int n_model = 5;
	HMM hmms[n_model];
	load_models(argv[1], hmms, n_model);
	FILE *seq = open_or_die(argv[2], "r");
	FILE *output = open_or_die(argv[3], "w");
	
	// variables
	int N = hmms[0].state_num;
	int T = 0;			// length of each line read from seq
	int n_sample = 0;	// number of lines in seq
	int obs[MAX_SEQ];	// buffer for a line from seq
	
	// dynamically allocated variables
	double **delta = malloc_2d(N, MAX_SEQ);
	
	// loop through each line in testing data
	rewind(seq);
	while((T = get_obs(seq, obs))!=-1){
		n_sample++;
		
		// loop through all models
		int model_max = -1;
		double max_among_models = 0;
		for(int m=0; m<n_model; m++){
			HMM *hmm = &hmms[m];
			
			// Viterbi
			for(int i=0; i<N; i++)
				delta[i][0] = hmm->initial[i] * hmm->observation[obs[0]][i];
			for(int t=1; t<T; t++)
				for(int i=0; i<N; i++){
					double max = 0;
					for(int j=0; j<N; j++)
						max = fmax(max, delta[j][t-1] * hmm->transition[j][i]);
					delta[i][t] = max * hmm->observation[obs[t]][i];
			}
			
			// find maximum probability among all paths
			double max = 0;
			for(int i=0; i<N; i++)
				max = fmax(max, delta[i][T-1]);
			
			// update maximum among all models
			if(max>max_among_models){
				max_among_models = max;
				model_max = m;
			}
		}
		
		// store result
		fprintf(output, "%s %e\n",
			hmms[model_max].model_name, max_among_models);
	}
	
	// clean up
	free(delta);
	fclose(output);
	fclose(seq);
	
	// calc accuracy if the answer is given
	if(strcmp(argv[2], "../testing_data1.txt")==0){
		output = open_or_die(argv[3], "r");
		FILE *ans = open_or_die("../testing_answer.txt", "r");
		n_sample = 0;
		int correct = 0;
		char result[MAX_LINE];
		char answer[MAX_LINE];
		while(fscanf(output, "%s %*s", result)!=EOF
			&& fscanf(ans, "%s", answer)!=EOF){
			if(strcmp(result, answer)==0)
				correct++;
			n_sample++;
		}
		FILE *acc = open_or_die("acc.txt", "w");
		fprintf(acc, "%f\n", (float)correct/n_sample);
		fprintf(stdout, "%f\n", (float)correct/n_sample);
		fclose(acc);
		fclose(ans);
		fclose(output);
	}
	
	// calc accuracy on training set to check if overfitting occured
	if(test_training_set){
		output = open_or_die(argv[3], "r");
		n_sample = 0;
		int correct = 0;
		char result[MAX_LINE];
		while(fscanf(output, "%s %*s", result)!=EOF){
			if(strcmp(result, argv[4])==0)
				correct++;
			n_sample++;
		}
		fprintf(stdout, "%f\n", (float)correct/n_sample);
		fclose(output);
	}
	
	return 0;
}
