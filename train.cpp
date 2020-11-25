#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <mat.h>

#include "util/quantize.h"

using namespace std;
using namespace gmm;

#define DATATYPE double
#define DATASIZE 10000
#define MIXNUM 20
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
	int label = -1;		// Class index
	DATATYPE iterNum = 0;
	DATATYPE lastCost = 0;
	DATATYPE currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_mixNum];
	DATATYPE* next_means = new DATATYPE[m_mixNum*3];	// New model for reestimation
    
	while (loop)
	{
		memset(counts, 0, sizeof(int)*m_mixNum);
		memset(next_means, 0, sizeof(DATATYPE)*3*m_mixNum);

		lastCost = currCost;
		currCost = 0;

		// Classification
		for (int i = 0; i < N; i++)
		{
			for(int j = 0; j < 3; j++)
				x[j] = data[i*3+j];

			currCost += GetLabel(x, &label, k_means);

			counts[label]++;
			for (int d = 0; d < 3; d++)
			{
				next_means[label*3 + d] += x[d];
			}
		}
		currCost /= N;

		// Reestimation
		for (int i = 0; i < m_mixNum; i++)
		{
			if (counts[i] > 0)
			{
				for (int d = 0; d < 3; d++)
				{
					next_means[i*3 + d] /= counts[i];
				}
                for (int j = 0; j < 3; j++){
                    k_means[i*3+j] = next_means[i*3+j];
                }
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(lastCost - currCost) < ENDERROR * lastCost)
		{
			unchanged++;
		}
		if (iterNum >= MAXITERNUM || unchanged >= 3)
		{
			loop = false;
		}

		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
	}

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

    int* counts = new int[m_mixNum];
	DATATYPE* overMeans = new DATATYPE[3];	// Overall mean of training data
	for (int i = 0; i < m_mixNum; i++)
	{
		counts[i] = 0;
		m_priors[i] = 0;
		memcpy(&(m_means[i*3]), &(k_means[i*3]), sizeof(DATATYPE) * DIMNUM);
		memset(&(m_vars[i*3]), 0, sizeof(DATATYPE) * DIMNUM);
	}
	memset(overMeans, 0, sizeof(DATATYPE) * 3);
	memset(m_minVars, 0, sizeof(DATATYPE) * 3);

	DATATYPE* x = new DATATYPE[3];
	int label = -1;

	for (int i = 0; i < datasize; i++)
	{
		for(int j=0;j<3;j++)
			x[j]=data[i*3+j];
		label=Label[i];

		// Count each Gaussian
		counts[label]++;
		DATATYPE* m = &(k_means[label*3]);
		for (int d = 0; d < 3; d++)
		{
			m_vars[label*3 + d] += (x[d] - m[d]) * (x[d] - m[d]);
		}

		// Count the overall mean and variance.
		for (int d = 0; d < 3; d++)
		{
			overMeans[d] += x[d];
			m_minVars[d] += x[d] * x[d];
		}
	}

	// Compute the overall variance (* 0.01) as the minimum variance.
	for (int d = 0; d < 3; d++)
	{
		overMeans[d] /= datasize;
		m_minVars[d] = max(MIN_VAR, (m_minVars[d] / datasize - overMeans[d] * overMeans[d])/100);
	}

	// Initialize each Gaussian.
	for (int i = 0; i < m_mixNum; i++)
	{
		m_priors[i] = 1.0 * counts[i];
		// / datasize;

		if (m_priors[i] > 0)
		{
			for (int d = 0; d < 3; d++)
			{
				m_vars[i*3 + d] = m_vars[i*3 + d] / counts[i];

				// A minimum variance for each dimension is required.
				if (m_vars[i*3 + d] < m_minVars[d])
				{
					m_vars[i*3 + d] = m_minVars[d];
				}
			}
		}
		else
		{
			memcpy(&(m_vars[i*3]), m_minVars, sizeof(DATATYPE) * 3);
			cout << "[WARNING] Gaussian " << i << " of GMM is not used!\n";
		}
	}
#ifdef FIX
	int position = floor(cpu_fix_pos_overflow(MIXNUM, m_priors, QUANTIZE_BIT));
	cpu_fix(MIXNUM, m_priors, m_priors, QUANTIZE_BIT, position);
#endif
	delete k_means;
	delete[] counts;
	delete[] overMeans;
	delete[] Label;

    bool loop = true;
	DATATYPE iterNum = 0;
	DATATYPE lastL = 0;
	DATATYPE currL = 0;
	int unchanged = 0;
    
	DATATYPE* next_priors = new DATATYPE[m_mixNum];
	DATATYPE* next_vars = new DATATYPE[m_mixNum*3];
	DATATYPE* next_means = new DATATYPE[m_mixNum*3];

	while (loop)
	{
		// Clear buffer for reestimation
		memset(next_priors, 0, sizeof(DATATYPE) * m_mixNum);
		memset(next_vars, 0, sizeof(DATATYPE) * m_mixNum * 3);
		memset(next_means, 0, sizeof(DATATYPE) * m_mixNum * 3);

		lastL = currL;
		currL = 0;

		// Predict
		for (int k = 0; k < datasize; k++)
		{
			for(int j=0;j<3;j++)
				x[j]=data[k*3+j];
			DATATYPE p = 0;
            for (int i = 0; i < m_mixNum; i++) {
		        p += m_priors[i] * GetProbability(x,  m_means, m_vars, i);
            }

			for (int j = 0; j < m_mixNum; j++)
			{
				DATATYPE pj = GetProbability(x, m_means, m_vars, j) * m_priors[j] / p;

				next_priors[j] += pj;

				for (int d = 0; d < 3; d++)
				{
					next_means[j*3+d] += pj * x[d];
					next_vars[j*3+d] += pj* x[d] * x[d];
				}
			}
			currL += (p > 1E-20) ? log10(p) : -20;
		}
#ifdef FIX
		cpu_fix(MIXNUM, next_priors, next_priors, QUANTIZE_BIT, position);
#endif
		currL /= datasize;

		// Reestimation: generate new priors, means and variances.
		for (int j = 0; j < m_mixNum; j++)
		{
			m_priors[j] = next_priors[j];
			// / datasize;

			if (m_priors[j] > 0)
			{
				for (int d = 0; d < 3; d++)
				{
					m_means[j*3+d] = next_means[j*3+d] / next_priors[j];
					m_vars[j*3+d] = next_vars[j*3+d] / next_priors[j] - m_means[j*3+d] * m_means[j*3+d];
					if (m_vars[j*3+d] < m_minVars[d])
					{
						m_vars[j*3+d] = m_minVars[d];
					}
				}
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(currL - lastL) < ENDERROR * fabs(lastL))
		{
			unchanged++;
		}
		if (iterNum >= MAXITERNUM || unchanged >= 3)
		{
			loop = false;
		}

		//--- Debug ---
		//cout << "Iter: " << iterNum << ", Average Log-Probability: " << currL << endl;
	}
	cout<<iterNum<<endl;
	delete[] next_priors;
	delete[] next_means;
	delete[] next_vars;
	delete[] x;
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

	MATFile *pmatFile_means = NULL;
    mxArray *pMxArray_means = NULL;
	MATFile *pmatFile_vars = NULL;
    mxArray *pMxArray_vars = NULL;

    const int datasize = 10000;     //Number of samples
    const int dim = 3;           //Dimension of feature
    //const int cluster_num = 4;   //Cluster number
	int split_index[6];
	fp >> tmp;
    int index = 0;
	for(int i =0; i < 180000; i++){
        fp >> tmp;
		if(i%(180000/datasize) == 0){
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
	cout<<"Fix"<<endl;
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

	for(int i =0; i < 10; i++){
		cout << m_priors[i] << endl;
		cout << m_means[i*3] << ' '<< m_means[i*3+1] << ' '<< m_means[i*3+2] << endl;
		cout << m_vars[i*3] << ' '<< m_vars[i*3+1] << ' '<< m_vars[i*3+2] << endl;
	}

	// write .mat	
	pmatFile_means = matOpen("means_1.mat","w");
	pMxArray_means = mxCreateDoubleMatrix(3, 10, mxREAL);
	mxSetData(pMxArray_means, m_means);
	matPutVariable(pmatFile_means, "means", pMxArray_means);
    matClose(pmatFile_means);

	pmatFile_vars = matOpen("vars_1.mat","w");
	pMxArray_vars = mxCreateDoubleMatrix(3, 10, mxREAL);
	mxSetData(pMxArray_vars, m_vars);
	matPutVariable(pmatFile_vars, "vars", pMxArray_vars);
    matClose(pmatFile_vars);

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