#ifndef MAIN_H_
#define MAIN_H_

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

//---Prototypes---
/**
 * @brief:
 * 1: Sigmoid
 * 2: Linear
 * */
vector<vector<float>> NNLayer(vector<vector<float>> input, vector<vector<float>> weight, vector<vector<float>> bias, uint8_t activation_function);
vector<vector<float>> Normalization(vector<vector<float>> in, float in_min, float in_max);
vector<vector<float>> DeNormalization(vector<vector<float>> in, float in_min, float in_max);

//---Variables---
vector<vector<float>> input = {
    {5.0},
    {-5.0},
    {0.0},
};
vector<vector<float>> normalized_input;
vector<vector<float>> denormalized_output;

vector<vector<float>> output_sigmoid;
vector<vector<float>> output_linear;

vector<vector<float>> weight1 = {
    {2.2693, 0.5865, 2.3852},
    {-0.5192, 2.0259, 1.1233},
    {1.3964, -3.0309, 0.5255},
    {-0.5526, 0.8239, -0.4767}};

vector<vector<float>> bias1 = {
    {1.4758},
    {1.1246},
    {1.3421},
    {0.2367},
};

vector<vector<float>> bias2 = {
    {-0.9249},
    {0.2414}};
vector<vector<float>> weight2 = {
    {-2.1927, 1.8469, 2.7873, -0.4511},
    {-1.5336, 1.4998, -0.3008, 0.9795}};

vector<vector<float>> NNLayer(vector<vector<float>> input, vector<vector<float>> weight, vector<vector<float>> bias, uint8_t activation_function)
{
    // get the col and row size of the input
    int m = input.size();
    int n = input[0].size();
    int p = weight.size();
    int q = weight[0].size();

    vector<vector<float>> output(p, vector<float>(n, 0));
    // multi dimentional matrix multiplications
    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < q; k++)
            {
                output[i][j] += weight[i][k] * input[k][j];
            }
            output[i][j] += bias[i][0];
            if (activation_function == 1)
            {
                output[i][j] = 1 / (1 + exp(-output[i][j]));
            }
            else if (activation_function == 2)
            {
                output[i][j] = output[i][j];
            }
        }
    }
    return output;
}

vector<vector<float>> Normalization(vector<vector<float>> in, float in_min, float in_max)
{
    // Make a buffer output mat (unnecessary, only for clean code for easier debugging)
    vector<vector<float>> out(in.size(), vector<float>(in[0].size(), 0));
    for (int i = 0; i < in.size(); i++)
    {
        for (int j = 0; j < in[i].size(); j++)
        {
            out[i][j] = 2 * (in[i][j] - in_min) / (in_max - in_min) - 1;
        }
    }
    return out;
}

vector<vector<float>> DeNormalization(vector<vector<float>> in, float in_min, float in_max)
{
    for (int i = 0; i < in.size(); i++)
    {
        for (int j = 0; j < in[i].size(); j++)
        {
            in[i][j] = 0.5 * (in[i][j] + 1) * (in_max - in_min) + in_min;
        }
    }
    return in;
}

#endif