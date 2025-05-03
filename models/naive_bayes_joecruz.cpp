#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

struct Sample
{
    vector<string> features;
    string label;
};

class NaiveBayesClassifier
{
private:
    map<string, int> classCounts;
    map<string, vector<map<string, int>>> featureCounts;
    set<string> labels;
    int numFeatures = 0;
    int totalSamples = 0;

    double calculateLogProbability(const string &label, const vector<string> &input) const
    {
        double logProb = log((double)classCounts.at(label) / totalSamples);
        for (int i = 0; i < numFeatures; ++i)
        {
            const auto &featureMap = featureCounts.at(label)[i];
            int count = featureMap.count(input[i]) ? featureMap.at(input[i]) : 0;

            int total = 0;
            for (const auto &p : featureMap)
            {
                total += p.second;
            }

            double prob = (count + 1.0) / (total + featureMap.size());
            logProb += log(prob);
        }
        return logProb;
    }

public:
    void train(const vector<Sample> &data)
    {
        if (data.empty())
            return;

        numFeatures = data[0].features.size();
        totalSamples = data.size();
        classCounts.clear();
        featureCounts.clear();
        labels.clear();

        for (const auto &sample : data)
        {
            classCounts[sample.label]++;
            labels.insert(sample.label);
        }

        for (const auto &label : labels)
        {
            featureCounts[label] = vector<map<string, int>>(numFeatures);
        }

        // Count feature frequencies for each class
        for (const auto &sample : data)
        {
            for (int i = 0; i < numFeatures; ++i)
            {
                featureCounts[sample.label][i][sample.features[i]]++;
            }
        }
    }

    string predict(const vector<string> &input) const
    {
        string bestLabel;
        double bestProb = -INFINITY;

        for (const auto &label : labels)
        {
            double logProb = calculateLogProbability(label, input);
            if (logProb > bestProb)
            {
                bestProb = logProb;
                bestLabel = label;
            }
        }

        return bestLabel;
    }
};

vector<Sample> loadData(const string &filename)
{
    vector<Sample> data;
    ifstream file(filename);
    string line;

    while (getline(file, line))
    {
        stringstream ss(line);
        string token;
        Sample sample;

        for (int i = 0; i < 3; ++i)
        {
            getline(ss, token, ',');
            sample.features.push_back(token);
        }

        getline(ss, token, ',');
        sample.label = token;
        data.push_back(sample);
    }

    return data;
}

void evaluateAccuracy(const NaiveBayesClassifier &nb, const vector<Sample> &testData)
{
    int correct = 0;
    for (const auto &sample : testData)
    {
        if (nb.predict(sample.features) == sample.label)
        {
            ++correct;
        }
    }
    double accuracy = (double)correct / testData.size() * 100.0;
    cout << "Accuracy on test data: " << accuracy << "%" << endl;
}

void splitAndEvaluate(NaiveBayesClassifier &nb, vector<Sample> &allData, double trainRatio)
{
    random_device rd;
    mt19937 g(rd());
    shuffle(allData.begin(), allData.end(), g);

    size_t trainSize = static_cast<size_t>(allData.size() * trainRatio);
    vector<Sample> trainingData(allData.begin(), allData.begin() + trainSize);
    vector<Sample> testData(allData.begin() + trainSize, allData.end());

    cout << "\nUsing " << trainRatio * 100 << "% of the data for training (" << trainingData.size() << " samples)" << endl;
    nb.train(trainingData);
    evaluateAccuracy(nb, testData);
}

int main()
{
    NaiveBayesClassifier nb;

    const string trainingFile = "data/bayes_training_data.txt";
    const string testFile = "data/bayes_test_data.txt";

    vector<Sample> trainingData = loadData(trainingFile);
    vector<Sample> testData = loadData(testFile);

    vector<double> trainRatios = {0.4, 0.6, 0.8, 1.0};
    for (double ratio : trainRatios)
    {
        size_t size = static_cast<size_t>(trainingData.size() * ratio);
        vector<Sample> subset(trainingData.begin(), trainingData.begin() + size);

        cout << "\nUsing " << ratio * 100 << "% of the training data (" << subset.size() << " samples)" << endl;
        nb.train(subset);
        evaluateAccuracy(nb, testData);
    }

    return 0;
}
