#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>

#include "util/quantize.h"

using namespace std;
using namespace gmm;

#define DATATYPE float
#define DATASIZE 200
#define MIXNUM 8
#define MAXITERNUM 100
#define ENDERROR 0.001
#define DIMNUM 3
#define QUANTIZE_BIT 9

DATATYPE data[30000] = {1};

DATATYPE CalcDistance(const DATATYPE* x, const DATATYPE* u) {
	DATATYPE temp = 0;
	for (int d = 0; d < 3; d++)
	{
		temp += (x[d] - u[d]) * (x[d] - u[d]);
	}
	return sqrt(temp);
}

DATATYPE GetLabel(const DATATYPE* sample, int* label, const DATATYPE* m_means) {
	DATATYPE dist = -1;
	for (int i = 0; i < MIXNUM; i++)
	{
		DATATYPE temp = CalcDistance(sample, &(m_means[i*3]));
		if (temp < dist || dist == -1)
		{
			dist = temp;
			*label = i;
		}
	}
	return dist;
}

void Cluster(DATATYPE* data, int N, int* Label, DATATYPE* k_means, int m_mixNum) {

	//assert(size >= m_clusterNum);

	// Recursion
	DATATYPE* x = new DATATYPE[3];	// Sample data
    DATATYPE* xt = new DATATYPE[3];
	int label = -1;		// Class index
	DATATYPE iterNum = 0;
	DATATYPE lastCost = 0;
	DATATYPE currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_mixNum];
	DATATYPE* next_means = new DATATYPE[m_mixNum*3];	// New model for reestimation
    int loop_Stall = log2(MIXNUM) + 1;

    struct compare_node{
        int index;
        DATATYPE num;
    };
    
	for (int loop=0; loop<16; loop++){
		memset(counts, 0, sizeof(int)*m_mixNum);
		memset(next_means, 0, sizeof(DATATYPE)*3*m_mixNum);

		lastCost = currCost;
		currCost = 0;
        DATATYPE local_dis[MIXNUM];
        compare_node first_layer[8];
        compare_node second_layer[4];
        compare_node third_layer[2];
        compare_node fourth_layer[1];

        DATATYPE one_hot[MIXNUM];
        DATATYPE sample_cache[loop_Stall][3];

        for(int i=0; i<DATASIZE+loop_Stall; i++) {
        //for(int i=0; i<6; i++) {
            if(i<DATASIZE){
                for(int j = 0; j < 3; j++)
                    x[j] = data[i*3+j];
                local_dis[0] = CalcDistance(x, k_means);
                local_dis[1] = CalcDistance(x, &(k_means[3]));
                local_dis[2] = CalcDistance(x, &(k_means[6]));
                local_dis[3] = CalcDistance(x, &(k_means[9]));
                local_dis[4] = CalcDistance(x, &(k_means[12]));
                local_dis[5] = CalcDistance(x, &(k_means[15]));
                local_dis[6] = CalcDistance(x, &(k_means[18]));
                local_dis[7] = CalcDistance(x, &(k_means[21]));
            }

            if(i>=loop_Stall) {
                for(int j=0; j<MIXNUM; j++){
                    one_hot[j] = 0;
                }
                for(int j=0; j<3; j++){
                    xt[j] = sample_cache[loop_Stall-1][j];
                }
                //cout<<xt[0]<<' '<<xt[1]<<' '<<xt[2]<<endl;
                one_hot[fourth_layer[0].index] = 1;
                next_means[0*3+0] += one_hot[0]*xt[0];  next_means[0*3+1] += one_hot[0]*xt[1];  next_means[0*3+2] += one_hot[0]*xt[2];
                next_means[1*3+0] += one_hot[1]*xt[0];  next_means[1*3+1] += one_hot[1]*xt[1];  next_means[1*3+2] += one_hot[1]*xt[2];
                next_means[2*3+0] += one_hot[2]*xt[0];  next_means[2*3+1] += one_hot[2]*xt[1];  next_means[2*3+2] += one_hot[2]*xt[2];
                next_means[3*3+0] += one_hot[3]*xt[0];  next_means[3*3+1] += one_hot[3]*xt[1];  next_means[3*3+2] += one_hot[3]*xt[2];
                next_means[4*3+0] += one_hot[4]*xt[0];  next_means[4*3+1] += one_hot[4]*xt[1];  next_means[4*3+2] += one_hot[4]*xt[2];
                next_means[5*3+0] += one_hot[5]*xt[0];  next_means[5*3+1] += one_hot[5]*xt[1];  next_means[5*3+2] += one_hot[5]*xt[2];
                next_means[6*3+0] += one_hot[6]*xt[0];  next_means[6*3+1] += one_hot[6]*xt[1];  next_means[6*3+2] += one_hot[6]*xt[2];
                next_means[7*3+0] += one_hot[7]*xt[0];  next_means[7*3+1] += one_hot[7]*xt[1];  next_means[7*3+2] += one_hot[7]*xt[2];
                counts[fourth_layer[0].index]++;
            }

            //数据FIFO
            for(int j=loop_Stall-1; j>0; j--){
                sample_cache[j][0] = sample_cache[j-1][0]; sample_cache[j][1] = sample_cache[j-1][1]; sample_cache[j][2] = sample_cache[j-1][2];
            }
            sample_cache[0][0] = x[0];sample_cache[0][1] = x[1];sample_cache[0][2] = x[2];

            //比较树
            if(third_layer[0].num < third_layer[1].num){
                fourth_layer[0].index = third_layer[0].index;  fourth_layer[0].num = third_layer[0].num;
            } else {
                fourth_layer[0].index = third_layer[1].index;  fourth_layer[0].num = third_layer[1].num;
            }

            for(int j=0; j<2; j++){
                if(second_layer[j*2].num < second_layer[j*2+1].num) {
                    third_layer[j].index = second_layer[j*2].index;  third_layer[j].num = second_layer[j*2].num;
                } else {
                    third_layer[j].index = second_layer[j*2+1].index;  third_layer[j].num = second_layer[j*2+1].num;
                }
            }

            for(int j=0; j<4; j++){
                if(first_layer[j*2].num < first_layer[j*2+1].num) {
                    second_layer[j].index = first_layer[j*2].index;  second_layer[j].num = first_layer[j*2].num;
                } else {
                    second_layer[j].index = first_layer[j*2+1].index;  second_layer[j].num = first_layer[j*2+1].num;
                }
            }

            for(int j=0; j<8; j++){
                first_layer[j].index = j; first_layer[j].num = local_dis[j];
            }
        }
        
        for(int i=0; i<MIXNUM; i++) {
            if(counts[i]>0){
                for(int j=0; j<3; j++){
                    k_means[i*3+j] = next_means[i*3+j]/counts[i];
                }
            }
        }
        /*for(int i=0; i<MIXNUM; i++){
            cout<<k_means[i*3]<<' '<<k_means[i*3+1]<<' '<<k_means[i*3+2]<<endl;
        }
        cout<<endl;*/
	}
    /*for(int i=0; i<MIXNUM; i++){
        cout<<k_means[i*3]<<' '<<k_means[i*3+1]<<' '<<k_means[i*3+2]<<endl;
    }*/
	// Output the label file
	for (int i = 0; i < N; i++)
	{
		for(int j = 0; j < 3; j++)
				x[j] = data[i*3+j];
		GetLabel(x, &label,  k_means);
		Label[i] = label;
	}
    
	delete[] counts;
	delete[] x;
	delete[] next_means;
}

DATATYPE GetProbability(const DATATYPE* x, const DATATYPE* m_means, const DATATYPE* m_vars, int j)
{
	DATATYPE p = 1;
	for (int d = 0; d < 3; d++)
	{
		p *= 1 / sqrt(2 * 3.14159 * m_vars[j*3+d]);
		p *= exp(-0.5 * (x[d] - m_means[j*3+d]) * (x[d] - m_means[j*3+d]) / m_vars[j*3+d]);
	}
	return p;
}

void train(DATATYPE* data, int datasize, DATATYPE* m_priors, DATATYPE* m_means, DATATYPE* m_vars, DATATYPE* m_minVars, int m_mixNum) {

    //Init guassian parameters
    const DATATYPE MIN_VAR = 1E-10;
    int* Label = new int[datasize];     //labels of each sample
    //Init kmeans parameters
    DATATYPE* k_means = new DATATYPE[m_mixNum*3];
    DATATYPE* cluster_sample = new DATATYPE[3];
    for (int i = 0; i < m_mixNum; i++) {
		int select = i * datasize / m_mixNum;
		for(int j = 0; j < 3; j++)
			cluster_sample[j] = data[select*3+j];
		memcpy(&(k_means[i*3]), cluster_sample, sizeof(DATATYPE) * 3);
	}
	delete[] cluster_sample;

    Cluster(data, datasize, Label, k_means, m_mixNum);
}

DATATYPE string_to_num(string str) {
	int i=0,len=str.length();
    DATATYPE sum=0;
	if(str[0] == '-'){
		i++;
	}
    while(i<len){
        if(str[i]=='.') break;
        sum=sum*10+str[i]-'0';
        ++i;
    }
    ++i;
    DATATYPE t=1,d=1;
    while(i<len){
        d*=0.1;
        t=str[i]-'0';
        sum+=t*d;
        ++i;
    }
	if(str[0] == '-'){
		sum *= -1;
	}
    return sum;
}

int main() {

	fstream fp;
	fp.open("../PointCloud6.csv");
	string tmp;

    const int datasize = 200;     //Number of samples
    const int dim = 3;           //Dimension of feature
    //const int cluster_num = 4;   //Cluster number
	int split_index[6];
	fp >> tmp;
    int index = 0;
	for(int i =0; i < 180000; i++){
        fp >> tmp;
		if(i%900 == 0){
			int k = 0;
			for(int j = 0; j < tmp.length(); j++) {
				if(tmp[j] == ',') {
					split_index[k] = j;
					k += 1;
				}
			}
			data[index*3] = string_to_num(tmp.substr(split_index[0]+1,(split_index[1]-split_index[0]-1)));
			data[index*3+1] = string_to_num(tmp.substr(split_index[1]+1,(split_index[2]-split_index[1]-1)));
			data[index*3+2] = string_to_num(tmp.substr(split_index[2]+1,(split_index[3]-split_index[2]-1)));
            index += 1;
		}
	}

#ifdef FIX
	int p = floor(cpu_fix_pos_overflow(DATASIZE, data, QUANTIZE_BIT));
	cpu_fix(DATASIZE, data, data, QUANTIZE_BIT, p);
#endif

//set gaussian parameters
    int m_dimNum = dim;        //Dimension of guassian feature
    int m_mixNum = MIXNUM;     //Number of guassian models

    int m_maxIterNum = MAXITERNUM;
    DATATYPE m_endError = ENDERROR;

    DATATYPE* m_priors;	// GaussianȨ权重
	DATATYPE* m_means;	// Gaussian均值
	DATATYPE* m_vars;	// Gaussian方差
    DATATYPE* m_minVars;

    m_priors = new DATATYPE[m_mixNum];
	m_means = new DATATYPE[m_mixNum*m_dimNum];
	m_vars = new DATATYPE[m_mixNum*m_dimNum];
    m_minVars = new DATATYPE[m_dimNum];

	train(data, datasize, m_priors, m_means, m_vars, m_minVars, m_mixNum);

	/*for(int i =0; i < MIXNUM; i++){
		cout << m_priors[i] << endl;
		cout << m_means[i*3] << ' '<< m_means[i*3+1] << ' '<< m_means[i*3+2] << endl;
		cout << m_vars[i*3] << ' '<< m_vars[i*3+1] << ' '<< m_vars[i*3+2] << endl;
	}*/

	fp.close();

#ifdef FIX
	/*DATATYPE x[100];
	DATATYPE y[100];
	for(int i = 0; i<100; i++) {
		x[i] = data[i];
	}

	int p = floor(cpu_fix_pos_overflow(100, x, 9));

	cpu_fix(100, x, x, 9, p);

	for(int i = 0; i<10; i++) {
		cout<< x[i] << ' '<< y[i] <<endl;
	}*/
#endif
    return 0;
}