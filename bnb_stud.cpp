#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <map>


#include "ts_stud.cpp"
// #include "greedy.cpp"
#include <ilcplex/ilocplex.h>
// #include "/opt/ibm/ILOG/CPLEX_Studio_Community221/cplex/include/ilcplex/ilocplex.h"

using namespace std;


class BnBSolver
{
public:

    static int GetRandom(int a, int b)
    {
        static mt19937 generator;
        uniform_int_distribution<int> uniform(a, b);
        return uniform(generator);
    }

    void find_ind_set(vector<int> cur_set, vector<int> &unused, map<pair<int, int>, bool> &pairs, 
                          vector<vector<int>> &indep_sets) {
        if (pairs.empty())
            return;

        bool found_large = false; // when no candidates in unused vector (leaf in recursion tree)

        auto i = unused.begin();
        int depth = cur_set.size();
        while(i != unused.end() && cur_set.size() < neighbours.size() ) {
            bool independent = true;
            for (int &j: cur_set) {
                if (neighbours[j].find(*i) != neighbours[j].end() || *i == j) {
                    independent = false;
                    break;
                }
            }
            if (independent) {
                int val = *i;
                cur_set.push_back(val);
                unused.erase(i);
                find_ind_set(cur_set, unused, pairs, indep_sets);
                found_large = true;    // To add only unique sets without their subsets
                cur_set.pop_back();
                i = unused.begin();
            } else {
                ++i;
            }
        }
        if (depth >= 3 && (depth == 1000 || !found_large)) {
            int c = 0;
            for (auto &i: cur_set)
                for (auto &j: cur_set)
                    if (pairs.find({min(i, j), max(i, j)}) != pairs.end()) {
                        pairs.erase({min(i, j), max(i, j)});
                        ++c;
                    }
            if (c != 0)
                indep_sets.push_back(cur_set);
        }
    }

    vector<vector<int>> find_all_sets() {
        map<pair<int, int>, bool> pairs; //no match for call to ‘(const std::hash<std::pair<int, int> >) in case of using unordered sets

        for (int i = 0; i < neighbours.size(); ++i) {
            for (int j = i + 1; j < neighbours.size(); ++j) {
                if (neighbours[i].find(j) == neighbours[i].end()) {
                    pairs.insert({{i, j}, true});
                }
            }
        }
        // cout << "Number of pairs: " << pairs.size() << endl;
        vector<vector<int>> indep_sets;
        for (int i = 0; i < neighbours.size(); ++i) {
            for (int _ = 0; _ < 5; ++_) {
                vector<int> unused;
                for (int j = 0; j < neighbours.size(); ++j) {
                    int vertex = GetRandom(0, neighbours.size() - 1);
                    if (vertex != i)
                        unused.push_back(vertex);
                }
                find_ind_set(vector<int>({i}), unused, pairs, indep_sets);
            }
        }
        cout << "Number of independent sets: " << indep_sets.size() << endl;
        for (auto &i : pairs)
            indep_sets.push_back(vector<int>({i.first.first, i.first.second}));
        return indep_sets;
    }

    void ReadGraphFile(string filename)
    {
        ifstream fin(filename);
        string line;
        file = filename;
        int vertices = 0, edges = 0;
        while (getline(fin, line))
        {
            if (line[0] == 'c')
            {
                continue;
            }
            stringstream line_input(line);
            char command;
            if (line[0] == 'p')
            {
                string in;
                line_input >> command >> in >> vertices >> edges;
                neighbours.resize(vertices);
            }
            else 
            {
                int st, fn;
                line_input >> command >> st >> fn;
                neighbours[st - 1].insert(fn - 1);
                neighbours[fn - 1].insert(st - 1);
            }
        }
        cout << "vertices : " << vertices << endl;
        model = IloModel(env);
        x = IloFloatVarArray(env, vertices);
        for (auto i = 0; i < vertices; ++i) {
            x[i] = IloFloatVar(env, 0., 1.);
        }
        IloRangeArray independent_sets(env);
        IloExpr expr(env);
        vector<vector<int>> sets = find_all_sets();
        for (const auto &set : sets) {
            for (auto &i : set)
                expr += x[i];
            independent_sets.add(IloRange(env, 0., expr, 1.));
            expr.clear();
        }
        model.add(independent_sets);
        for (int i = 0; i < vertices; ++i) 
            expr += x[i];

        IloObjective obj(env, expr, IloObjective::Maximize);
        model.add(obj);
        expr.end();
    }

    pair<double, vector<float>> solve_cplex(vector<int>& candidates) {
        if (candidates.empty())
            return {(double) clique.size(), {}};
        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());

        double ans = -1;
        vector<float> variables;
        if (cplex.solve()) {
            size_t largest_x = candidates.size() - 1;
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (cplex.getValue(x[candidates[i]]) > cplex.getValue(x[candidates[largest_x]])) {
                    largest_x = i;
                }
            }
            if (largest_x != candidates.size() - 1)
                swap(candidates[candidates.size() - 1], candidates[largest_x]); // in recursion we always try to add the last element from candidates vector
            ans = cplex.getObjValue();
            for (int i = 0; i < neighbours.size(); ++i) {
                variables.push_back(cplex.getValue(x[i]));
            }
        }
        cplex.end();
        return {ans, variables};
    }

    void RunBnB()
    {
        MaxCliqueTabuSearch st;
        st.ReadGraphFile(file);
        st.RunSearch(neighbours.size() * 10, 7);
        best_clique = st.GetClique();
        cout << "Tabu Search Initial Clique : " << best_clique.size() << endl;
        vector<int> candidates;
        for (int i = 0; i < neighbours.size(); ++i)
        {
            candidates.push_back(i);
        }
        static mt19937 generator;
        shuffle(candidates.begin(), candidates.end(), generator);
        pair<int, int> prev_vertex = {-1, 0};
        // cout << "Candidates size : " << candidates.size() << endl;
        start = clock();
        BnBRecursion(candidates, prev_vertex);
    }

    const unordered_set<int>& GetClique()
    {
        return best_clique;
    }

    bool Check()
    {
        for (int i : clique)
        {
            for (int j : clique)
            {
                if (i != j && neighbours[i].count(j) == 0)
                {
                    cout << "Returned subgraph is not clique\n";
                    return false;
                }
            }
        }
        return true;
    }

    void ClearClique()
    {
        best_clique.clear();
        clique.clear();
    }

    IloModel model;
    IloEnv env;
    IloFloatVarArray x;

private:
    bool isIntValues(vector<float>& values) {
        for (float val: values) {
            if ((val - EPS > 0) && (val + EPS < 1)) 
                return false;
        }
        return true;
    }
    void BnBRecursion(vector<int>& candidates, pair<int, int> prev_vertex)
    {
        if (clique.size() + candidates.size() <= best_clique.size())
            return;
        if ((double)((clock() - start) / CLOCKS_PER_SEC) > 2000.) 
            return;\
        pair<int, int> prev_vertex_copy = prev_vertex;
        bool prev_vertex_in = (prev_vertex_copy.second == 1);
        bool is_first_vertex = (prev_vertex_copy.first == -1);

        IloRange constraint;
        IloExpr expr(env);
        if (!is_first_vertex) {
            if (prev_vertex_in)
                clique.insert(prev_vertex_copy.first);
            expr = x[prev_vertex_copy.first];
            constraint = IloRange(env, prev_vertex_copy.second, expr, prev_vertex_copy.second);
            model.add(constraint);
        }

        auto res = solve_cplex(candidates); // res.first - obj func, res.second - x[i] (cplex variables)
        double sol = res.first;

        if (floor(sol + EPS) <= clique.size() || floor(sol + EPS) <= best_clique.size()) {
            if (best_clique.size() < clique.size()) {
                best_clique  = clique;
                cout << "New best clique : " << best_clique.size() << endl;
            }
            if (prev_vertex_in) 
                clique.erase(prev_vertex_copy.first);
            if (!is_first_vertex)
                model.remove(constraint);
            expr.end();
            return;
        }
        if (isIntValues(res.second)) {
            if (clique.size() > best_clique.size()) {
                best_clique = clique;
                // cout << "New best clique : " << best_clique.size() << endl;
            }
        }
        if (candidates.empty()) {
            if (clique.size() > best_clique.size()) {
                best_clique = clique;
                cout << "New best clique : " << best_clique.size() << endl;
            }
            if (prev_vertex_in)
                clique.erase(prev_vertex_copy.first);
            if (!is_first_vertex)
                model.remove(constraint);

            expr.end();
            return;
        }
        int vertex = candidates.back();
        candidates.pop_back();

        bool all_edges = true;
        for (int i: clique) {
            if (neighbours[vertex].find(i) == neighbours[vertex].end()) {
                all_edges = false;
                break;
            }
        }
        if (all_edges) {
            prev_vertex.first = vertex;
            prev_vertex.second = 1;
            BnBRecursion(candidates, prev_vertex);
        }

        prev_vertex.first = vertex;
        prev_vertex.second = 0;
        BnBRecursion(candidates, prev_vertex);

        candidates.push_back(vertex);
        if (!is_first_vertex) {
            model.remove(constraint);
            if (prev_vertex_in)
                clique.erase(prev_vertex_copy.first);
        }
        expr.end();
    }

private:
    double EPS = 0.0001;
    clock_t start;
    vector<unordered_set<int>> neighbours;
    pair<int, bool> prev_v;
    unordered_set<int> best_clique;
    unordered_set<int> clique;
    string file;
};

int main()
{
    // ios_base::sync_with_stdio(false);
    // cin.tie(nullptr);

    vector<string> files = {
        "brock200_1.clq", "brock200_2.clq", "brock200_3.clq", "brock200_4.clq",
        "c-fat200-1.clq", "c-fat200-2.clq", "c-fat200-5.clq", "c-fat500-1.clq",
                            "c-fat500-10.clq", "c-fat500-2.clq", "c-fat500-5.clq", 
                            "C125.9.clq", "gen200_p0.9_44.clq", "gen200_p0.9_55.clq", "johnson8-2-4.clq", "johnson8-4-4.clq",
                            "johnson16-2-4.clq",
                            "hamming6-2.clq", "hamming6-4.clq", "hamming8-2.clq", "hamming8-4.clq", 
                            "keller4.clq", "MANN_a9.clq", "MANN_a27.clq", "MANN_a45.clq", "p_hat300-1.clq",
                            "p_hat300-2.clq", "p_hat300-3.clq", "san200_0.7_1.clq", "san200_0.7_2.clq", 
                            "san200_0.9_1.clq", "san200_0.9_2.clq",
                            "san200_0.9_3.clq", "sanr200_0.7.clq"};
    // vector<string> files = {"MANN_a27.clq"};

    ofstream fout("clique_bnb.csv");
    fout << "File; Clique; Time (sec)\n";
    for (string file : files)
    {
#ifdef NDEBUG
        string filename = "max_clique_txt/DIMACS_all_ascii/" + file;
#else
        string filename = "../max_clique_txt/DIMACS_all_ascii/" + file;
#endif
        cout << "====== " << file << " ======" << endl;
        BnBSolver problem;
        problem.ClearClique();
        clock_t start = clock();
        problem.ReadGraphFile(filename);
        problem.RunBnB();
        double time = (double)(clock() - start) / CLOCKS_PER_SEC;
        if (! problem.Check())
        {
            cout << "*** WARNING: incorrect clique ***\n";
            fout << "*** WARNING: incorrect clique ***\n";
        }
        fout << file << "; " << problem.GetClique().size() << "; " << time << '\n';
        cout << file << ", result - " << problem.GetClique().size() << ", time - " << time << '\n';
    }
    return 0;
}
