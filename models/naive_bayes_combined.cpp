#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Sample {
    vector<string> features;
    string label;
};

class NaiveBayesClassifier {
private:
    map<string, int> classCounts;
    map<string, vector<map<string, int>>> featureCounts;
    set<string> labels;
    int numFeatures = 0;
    int totalSamples = 0;

public:
    void train(const vector<Sample>& data) {
        if (data.empty()) return;

        numFeatures = data[0].features.size();
        totalSamples = data.size();

        classCounts.clear();
        featureCounts.clear();
        labels.clear();

        for (const auto& sample : data) {
            classCounts[sample.label]++;
            labels.insert(sample.label);
        }

        for (const string& label : labels) {
            featureCounts[label] = vector<map<string, int>>(numFeatures);
        }

        for (const auto& sample : data) {
            for (int i = 0; i < numFeatures; ++i) {
                featureCounts[sample.label][i][sample.features[i]]++;
            }
        }
    }

    string predict(const vector<string>& input) const {
        string bestLabel;
        double bestLogProb = -INFINITY;

        for (const string& label : labels) {
            double logProb = log((double)classCounts.at(label) / totalSamples);

            for (int i = 0; i < numFeatures; ++i) {
                const auto& featureMap = featureCounts.at(label)[i];
                int count = featureMap.count(input[i]) ? featureMap.at(input[i]) : 0;

                int total = 0;
                for (const auto& entry : featureMap) {
                    total += entry.second;
                }

                double prob = (count + 1.0) / (total + featureMap.size()); //Laplce smoothing applied here
                logProb += log(prob);
            }

            if (logProb > bestLogProb) {
                bestLogProb = logProb;
                bestLabel = label;
            }
        }

        return bestLabel;
    }

    double evaluate(const vector<Sample>& testData) const {
        int correct = 0;
        for (const auto& sample : testData) {
            string predicted = predict(sample.features);
            cout << "Predicted: " << predicted << ", Actual: " << sample.label << "\n";
            if (predicted == sample.label) correct++;
        }
        double accuracy = static_cast<double>(correct) / testData.size();
        double errorRate = 1.0 - accuracy;


        cout << fixed << setprecision(2);
        cout << "\nAccuracy: " << accuracy * 100.0 << "%";
        cout << "\nError Rate: " << errorRate * 100.0 << "%\n";

        return accuracy;
    }
};

//For reducing whitespace
string trim(const string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, last - first + 1);
}

//Loading csv
vector<Sample> loadLabeledData(const string& filename) {
    vector<Sample> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        Sample sample;

        for (int i = 0; i < 3; ++i) { //assume 3 features, can be adjusted to accomadate more 
            getline(ss, token, ',');
            sample.features.push_back(trim(token));
        }

        if (getline(ss, token)) {
            sample.label = trim(token);
            data.push_back(sample);
        }
    }
    return data;
}

int main() {
    string trainingPath = "./data/bayes_training_data.txt";
    string testPath = "./data/bayes_test_data.txt";

    vector<Sample> trainingData = loadLabeledData(trainingPath);
    vector<Sample> testData = loadLabeledData(testPath);

    NaiveBayesClassifier classifier;
    classifier.train(trainingData);

    cout << "\n=== Predictions ===\n";
    double accuracy = classifier.evaluate(testData);

    return 0;
}
