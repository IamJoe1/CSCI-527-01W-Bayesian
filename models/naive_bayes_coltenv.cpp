#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
using namespace std;

// All possible values
const vector<string> eye_colors = {"red", "blue", "yellow"};
const vector<string> modes = {"truck", "car", "jet", "animal"};
const vector<string> skills = {"repair", "scout", "supply", "attack"};
const vector<string> factions = {"good", "evil"};

struct Record {
	string eye_color;
	string mode;
	string skill;
	string faction;
};

string trim(const string& str) {
	size_t first = str.find_first_not_of(" \n\r\t");
	if (first == string::npos) return "";
	size_t last = str.find_last_not_of(" \n\r\t");
	return str.substr(first, last - first + 1);
}

//Naive Bayes Classifier defintion
class NaiveBayes {
	map<string, int> class_counts;
	map<string, map<string, map<string, int>>> feature_counts;
	int total_records = 0;

public:
	void train(const string& filename) {
		ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            exit(1);
        }
		string line;
		while (getline(file, line)) {
			Record r = parse_record(line);
			class_counts[r.faction]++;
			feature_counts["eye_color"][r.eye_color][r.faction]++;
			feature_counts["mode"][r.mode][r.faction]++;
			feature_counts["skill"][r.skill][r.faction]++;
			total_records++;
		}

        
        
	}

	string predict(const Record& r) {
		double max_prob = -1;
		string best_class;
		for (const string& cls : factions) {
			double prob = log((double)class_counts[cls] / total_records); // prior
			prob += log(get_cond_prob("eye_color", r.eye_color, cls));
			prob += log(get_cond_prob("mode", r.mode, cls));
			prob += log(get_cond_prob("skill", r.skill, cls));
			if (prob > max_prob || best_class.empty()) {
				max_prob = prob;
				best_class = cls;
			}
		}
		return best_class;
	}

	void test(const string& filename) {
		ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            exit(1);
        }
		string line;
		int total = 0, correct = 0;
		while (getline(file, line)) {
			Record r = parse_record(line);
			string predicted = predict(r);
			cout << "Predicted: " << predicted << ", Actual: " << r.faction << endl;
			if (predicted == r.faction) correct++;
			total++;
		}
		cout << "Accuracy: " << (double)correct / total * 100 << "%" << endl;

       
	}

private:
	Record parse_record(const string& line) {
		stringstream ss(line);
		string item;
		Record r;
		getline(ss, r.eye_color, ',');
		getline(ss, r.mode, ',');
		getline(ss, r.skill, ',');
		getline(ss, r.faction);
		r.faction = trim(r.faction);
		return r;
	}

	double get_cond_prob(const string& attr, const string& value, const string& cls) {
		int count = feature_counts[attr][value][cls];
		int total_class = class_counts[cls];

		//Laplace smoothing here 
		int k = 1;
		int V = attr == "eye_color" ? eye_colors.size() :
		        attr == "mode" ? modes.size() :
		        attr == "skill" ? skills.size() : 0;

		return (count + k) / (double)(total_class + V);
	}
};

int main() {
	NaiveBayes nb;
	nb.train("./data/bayes_training_data.txt");
	nb.test("./data/bayes_test_data.txt");
	return 0;
}
