#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>



using namespace std;

struct Robot {
    string eye_color;
    string mode;
    string skill;
    string faction;
};

class NaiveBayesClassifier {
private:
    map<string, int> class_counts;
    map<string, map<string, map<string, int>>> feature_counts;
    int total_samples = 0;
    vector<string> classes = { "good", "evil" };

    vector<string> eye_colors = { "red", "blue", "yellow" };
    vector<string> modes = { "truck", "car", "jet", "animal" };
    vector<string> skills = { "repair", "scout", "supply", "attack" };

public:
    void train(const vector<Robot>& data) {
        for (const auto& robot : data) {
            class_counts[robot.faction]++;
            feature_counts["eye_color"][robot.faction][robot.eye_color]++;
            feature_counts["mode"][robot.faction][robot.mode]++;
            feature_counts["skill"][robot.faction][robot.skill]++;
            total_samples++;
        }
    }

    string predict(const Robot& robot) {
        double max_prob = -1.0;
        string best_class;

        for (const auto& cls : classes) {
            double prob = (double)(class_counts[cls] + 1) / (total_samples + classes.size());
            prob *= get_conditional_prob("eye_color", cls, robot.eye_color, eye_colors.size());
            prob *= get_conditional_prob("mode", cls, robot.mode, modes.size());
            prob *= get_conditional_prob("skill", cls, robot.skill, skills.size());

            if (prob > max_prob) {
                max_prob = prob;
                best_class = cls;
            }
        }
        return best_class;
    }

private:
    double get_conditional_prob(const string& feature, const string& cls, const string& value, int value_count) {
        int count = feature_counts[feature][cls][value] + 1;
        int total = class_counts[cls] + value_count;
        return (double)count / total;
    }
};

vector<Robot> load_data(const string& filepath, bool has_label) {
    vector<Robot> data;
    ifstream file(filepath);

    if (!file.is_open()) {
        cerr << "ERROR: Could not open file: " << filepath << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string eye, mode, skill, faction;

        getline(ss, eye, ',');
        getline(ss, mode, ',');
        getline(ss, skill, ',');

        if (has_label)
            getline(ss, faction, '\n');
        else
            faction = "";

        if (!eye.empty() && !mode.empty() && !skill.empty()) {
            data.push_back({ eye, mode, skill, faction });
        }
    }

    return data;
}

int main() {

    vector<Robot> training_data = load_data("./data/bayes_training_data.txt", true);
    vector<Robot> test_data = load_data("./data/bayes_test_data.txt", false);
    cout << "Test Data Loaded: " << test_data.size() << " samples\n";

    NaiveBayesClassifier classifier;
    classifier.train(training_data);

    cout << "Predictions on Test Data:\n";
    for (const auto& test_robot : test_data) {
        string prediction = classifier.predict(test_robot);
        cout << test_robot.eye_color << "," << test_robot.mode << "," << test_robot.skill
            << " => " << prediction << "\n";
    }


    return 0;
}