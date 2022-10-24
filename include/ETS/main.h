#ifndef MAIN_H_
#define MAIN_H_

#include <iostream>
#include <vector>
#include <math.h>
#include "opencv4/opencv2/opencv.hpp"
#include <opencv4/opencv2/highgui.hpp>

using namespace std;

//---Prototypes---
/**
 * @brief:
 * 1: Sigmoid
 * 2: Linear
 * */
vector<vector<float>> NNLayer(vector<vector<float>> input, vector<vector<float>> weight, vector<vector<float>> bias, uint8_t activation_function);

vector<vector<float>> Normalization(vector<vector<float>> in, float in_min, float in_max);

vector<vector<float>> NormalizationWithMat(cv::Mat in, float in_min, float in_max);

vector<vector<float>> DeNormalization(vector<vector<float>> in, float in_min, float in_max);

// make a function to random the weight and bias
void RandomWeight(vector<vector<float>> *weight, int row, int col);

void RandomBias(vector<vector<float>> *bias, int row, int col);

//---Variables---
vector<vector<float>> input;
vector<vector<float>> normalized_input;
vector<vector<float>> denormalized_output;

vector<vector<float>> output_sigmoid;
vector<vector<float>> output_linear;

vector<vector<float>> weight1;

vector<vector<float>> bias1;

vector<vector<float>> bias2;
vector<vector<float>> weight2;

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

vector<vector<float>> NormalizationWithMat(cv::Mat in, float in_min, float in_max)
{
    // Make a buffer output mat (unnecessary, only for clean code for easier debugging)
    vector<vector<float>> out(in.rows, vector<float>(in.cols, 0));
    for (int i = 0; i < in.rows; i++)
    {
        for (int j = 0; j < in.cols; j++)
        {
            out[i][j] = 2 * (in.at<float>(i, j) - in_min) / (in_max - in_min) - 1;
        }
    }
    return out;
}

void RandomWeight(vector<vector<float>> *weight, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            // print the weight
            (*weight)[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void RandomBias(vector<vector<float>> *bias, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            (*bias)[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

#endif