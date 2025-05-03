#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <set>
#include <cmath>

using namespace std;

/**
 *  Run:
 *      g++ -std=c++11 -o models/naive_bayes_joecruz models/naive_bayes_joecruz.cpp
 *      ./models/naive_bayes_joecruz
 *  */

struct Sample
{
    vector<string> features;
    string label;
};

class NaiveBayesClassifier

{
private:
    vector<Sample> trainingData;
    map<string, int> classCounts;
    map<string, vector<map<string, int>>> featureCounts;
    set<string> labels;
    int numFeatures = 0;

    double calculateLogProbability(const string &label, const vector<string> &input)
    {
        double logProb = log((double)classCounts[label] / trainingData.size());
        for (int i = 0; i < numFeatures; ++i)
        {
            int count = featureCounts[label][i][input[i]];
            int total = 0;
            for (const auto &p : featureCounts[label][i])
            {
                total += p.second;
            }
            double prob = (count + 1.0) / (total + featureCounts[label][i].size());
            logProb += log(prob);
        }
        return logProb;
    }

public:
    void train(const vector<Sample> &data)
    {
        trainingData = data;
        if (trainingData.empty())
            return;

        numFeatures = trainingData[0].features.size();

        for (const auto &sample : trainingData)
        {
            labels.insert(sample.label);
            classCounts[sample.label]++;
        }

        for (const auto &label : labels)
        {
            featureCounts[label] = vector<map<string, int>>(numFeatures);
        }

        for (const auto &sample : trainingData)
        {
            for (int i = 0; i < numFeatures; ++i)
            {
                featureCounts[sample.label][i][sample.features[i]]++;
            }
        }
    }

    string predict(const vector<string> &input)
    {
        string bestLabel;
        double bestProb = -1.0;

        for (const auto &label : labels)
        {
            double logProb = calculateLogProbability(label, input);
            if (bestLabel.empty() || logProb > bestProb)
            {
                bestProb = logProb;
                bestLabel = label;
            }
        }

        return bestLabel;
    }
};

vector<string> getUserInput(int numFeatures)
{
    cout << "Welcome to the Naive Bayes Classifier!" << endl;
    cout << "Features to choose from: sunny, rainy, cloudy, hot, mild, cold" << endl;
    cout << endl;
    vector<string> input;
    cout << "Enter the values for the features (ex: rainy cold): ";
    string feature;

    for (int i = 0; i < numFeatures; ++i)
    {
        cin >> feature;
        input.push_back(feature);
    }

    return input;
}

int main()
{
    NaiveBayesClassifier nb;
    vector<Sample> trainingData = {
        {{"sunny", "hot"}, "beach"},
        {{"rainy", "cold"}, "stay_home"},
        {{"cloudy", "mild"}, "park"},
        {{"sunny", "mild"}, "hike"},
        {{"rainy", "mild"}, "museum"},
        {{"cloudy", "cold"}, "stay_home"},
        {{"sunny", "cold"}, "skiing"}};

    nb.train(trainingData);

    vector<string> testInput = getUserInput(2);

    string result = nb.predict(testInput);
    cout << "Predicted class: " << result << endl;

    return 0;
}
