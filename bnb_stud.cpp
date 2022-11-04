#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <random>
#include <unordered_set>
#include <algorithm>

#include "ts_stud.cpp"
#include "/opt/ibm/ILOG/CPLEX_Studio_Community221/cplex/include/ilcplex/ilocplex.h"

using namespace std;

/*
class MaxCliqueTabuSearch
{
public:
    static int GetRandom(int a, int b)
    {
        static mt19937 generator;
        uniform_int_distribution<int> uniform(a, b);
        return uniform(generator);
    }

    void ReadGraphFile(string filename)
    {
        ifstream fin(filename);
        string line;
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
                string type;
                line_input >> command >> type >> vertices >> edges;
                neighbour_sets.resize(vertices);
                qco.resize(vertices);
                index.resize(vertices, -1);
                non_neighbours.resize(vertices);
            }
            else
            {
                int start, finish;
                line_input >> command >> start >> finish;
                // Edges in DIMACS file can be repeated, but it is not a problem for our sets
                neighbour_sets[start - 1].insert(finish - 1);
                neighbour_sets[finish - 1].insert(start - 1);
            }
        }
        for (int i = 0; i < vertices; ++i)
        {
            for (int j = 0; j < vertices; ++j)
            {
                if (neighbour_sets[i].count(j) == 0 && i != j)
                    non_neighbours[i].insert(j);
            }
        }
    }

    void RunSearch(int starts, int randomization)
    {
        for (int iter = 0; iter < starts; ++iter)
        {
            ClearClique();
            for (size_t i = 0; i < neighbour_sets.size(); ++i)
            {
                qco[i] = i;
                index[i] = i;
            }
            RunInitialHeuristic(randomization);
            c_border = q_border;
            int swaps = 0;
            while (swaps < 100)
            {
                if (! Move())
                {
                    if (! Swap1To1())
                    {
                        break;
                    }
                    else
                    {
                        ++swaps;
                    }
                }
            }
            if (q_border > best_clique.size())
            {
                best_clique.clear();
                for (int i = 0; i < q_border; ++i)
                    best_clique.insert(qco[i]);
            }
        }
    }

    const unordered_set<int>& GetClique()
    {
        return best_clique;
    }

    bool Check()
    {
        for (int i : best_clique)
        {
            for (int j : best_clique)
            {
                if (i != j && neighbour_sets[i].count(j) == 0)
                {
                    cout << "Returned subgraph is not a clique\n";
                    return false;
                }
            }
        }
        return true;
    }

    void ClearClique()
    {
        best_clique.clear();
        q_border = 0;
        c_border = 0;
    }

private:
    int ComputeTightness(int vertex)
    {
        int tightness = 0;
        for (int i = 0; i < q_border; ++i)
        {
            if (neighbour_sets[qco[i]].count(vertex) == 0)
                ++tightness;
        }
        return tightness;
    }

    void SwapVertices(int vertex, int border)
    {
        int vertex_at_border = qco[border];
        swap(qco[index[vertex]], qco[border]);
        swap(index[vertex], index[vertex_at_border]);
    }

    void InsertToClique(int i)
    {
        for (int j : non_neighbours[i])
        {
            if (ComputeTightness(j) == 0)
            {
                --c_border;
                SwapVertices(j, c_border);
            }
        }
        SwapVertices(i, q_border);
        ++q_border;
    }

    void RemoveFromClique(int k)
    {
        for (int j : non_neighbours[k])
        {
            if (ComputeTightness(j) == 1)
            {
                SwapVertices(j, c_border);
                c_border++;
            }
        }
        --q_border;
        SwapVertices(k, q_border);
    }

    bool Swap1To1()
    {
        int st = GetRandom(0, q_border - 1);
        for (int counter = 0; counter < q_border; ++counter)
        {
            int vertex_index = (counter + st) % q_border;
            int vertex = qco[vertex_index];
            vector<int> L;
            for (int i : non_neighbours[vertex])
            {
                if (ComputeTightness(i) == 1)
                {
                    L.push_back(i);
                }
            }
            if (L.empty())
                continue;
            int index_in_l = GetRandom(0, L.size() - 1);
            int change = L[index_in_l];
            RemoveFromClique(vertex);
            InsertToClique(change);
            return true;
        }
        return false;
    }

    bool Move()
    {
        if (c_border == q_border)
            return false;
        int index_in_qco = GetRandom(q_border, c_border - 1);
        int vertex = qco[index_in_qco];
        InsertToClique(vertex);
        return true;
    }

    void RunInitialHeuristic(int randomization)
    {
        static mt19937 generator;
        vector<int> candidates(neighbour_sets.size());
        for (size_t i = 0; i < neighbour_sets.size(); ++i)
        {
            candidates[i] = i;
        }
        shuffle(candidates.begin(), candidates.end(), generator);
        while (! candidates.empty())
        {
            int last = candidates.size() - 1;
            int rnd = GetRandom(0, min(randomization - 1, last));
            int vertex = candidates[rnd];
            SwapVertices(vertex, q_border);
            ++q_border;
            for (int c = 0; c < candidates.size(); ++c)
            {
                int candidate = candidates[c];
                if (neighbour_sets[vertex].count(candidate) == 0)
                {
                    // Move the candidate to the end and pop it
                    swap(candidates[c], candidates[candidates.size() - 1]);
                    candidates.pop_back();
                    --c;
                }
            }
            shuffle(candidates.begin(), candidates.end(), generator);
        }
    }

private:
    vector<unordered_set<int>> neighbour_sets;
    vector<unordered_set<int>> non_neighbours;
    unordered_set<int> best_clique;
    vector<int> qco;
    vector<int> index;
    int q_border = 0;
    int c_border = 0;
};
*/

class BnBSolver
{
public:
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
            if (line[0] == 'p')
            {
                stringstream s(line);
                char c;
                string in;
                s >> c >> in >> vertices >> edges;
                neighbours.resize(vertices);
                non_neighbours.resize(vertices);
            }
            else
            {
                stringstream s(line);
                char c;
                int st, fn;
                s >> c >> st >> fn;
                neighbours[st - 1].insert(fn - 1);
                neighbours[fn - 1].insert(st - 1);
            }
        }
        for (int i = 0; i < vertices; ++i)
        {
            for (int j = 0; j < vertices; ++j)
            {
                if (neighbour_sets[i].count(j) == 0 && i != j)
                    non_neighbours[i].insert(j);
            }
            tightness[i] = 0;
        }
        model = IloModel(env);
        x = IloFloatVarArray(env, neighbours.size());
        // Create cplex variable for each node
        for (size_t i = 0; i < vertices; ++i)
            x[i] = IloFloatVar(env, 0., 1.);
        
        // Create expressions for all
        IloRangeArray i_sets(env);
        IloExpr expr(env);
        for (const auto &set: non_neighbours) {
            for (auto &i: set)
                expr += x[i];
            i_sets.add(IloRange(env, 0., expr, 1.));
            expr.clear();
        }
    }

    void RunBnB()
    {
        MaxCliqueTabuSearch st;
        st.ReadGraphFile(file);
        st.RunSearch(5, 10);
        best_clique = st.GetClique();
        vector<int> candidates(neighbours.size());
        for (size_t i = 0; i < neighbours.size(); ++i)
        {
            candidates[i] = i;
        }
        static mt19937 generator;
        shuffle(candidates.begin(), candidates.end(), generator); // TODO: Pardalos's order
        BnBRecursion(candidates);
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
    void BnBRecursion(const vector<int>& candidates)
    {
        if (candidates.empty())
        {
            if (clique.size() > best_clique.size())
            {
                best_clique = clique;
            }
            return;
        }

        if (clique.size() + candidates.size() <= best_clique.size()) // TODO: Upper bound (Fast), add more
            return;

        for (size_t c = 0; c < candidates.size(); ++c)
        {
            vector<int> new_candidates;
            new_candidates.reserve(candidates.size());
            for (size_t i = c + 1; i < candidates.size(); ++i)  // Filtering the candidates
            {
                if (neighbours[candidates[c]].count(candidates[i]) != 0)
                    new_candidates.push_back(candidates[i]);
            }
            clique.insert(candidates[c]);
            BnBRecursion(new_candidates);
            clique.erase(candidates[c]);
        }
    }


private:
    vector<unordered_set<int>> neighbours;
    vector<unordered_set<int>> non_neighbours;
    unordered_set<int> best_clique;
    unordered_set<int> clique;
    string file;
};

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<string> files = { /*"C125.9.clq",*/ "johnson8-2-4.clq", "johnson16-2-4.clq", "MANN_a9.clq", /*"MANN_a27.clq",
        "p_hat1000-1.clq",*/ "keller4.clq", "hamming8-4.clq", /*"brock200_1.clq",*/ "brock200_2.clq", "brock200_3.clq", "brock200_4.clq",
        /*"gen200_p0.9_44.clq", "gen200_p0.9_55.clq", "brock400_1.clq", "brock400_2.clq", "brock400_3.clq", "brock400_4.clq",
        "MANN_a45.clq", "sanr400_0.7.clq", "p_hat1000-2.clq", "p_hat500-3.clq", "p_hat1500-1.clq", "p_hat300-3.clq", "san1000.clq",
        "sanr200_0.9.clq"*/ };
    ofstream fout("clique_bnb.csv");
    fout << "File; Clique; Time (sec)\n";
    for (string file : files)
    {
        BnBSolver problem;
        string filename = "max_clique_txt/DIMACS_all_ascii/" + file;
        problem.ReadGraphFile(filename);
        problem.ClearClique();
        clock_t start = clock();
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
