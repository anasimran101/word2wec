#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vocab.h>
#include <vector>
#include <unordered_map>
#include <map>


std::unordered_map<std::string, int> vocab_hash;
std::vector<word> vocab_list;

std::string train_corpus_file, output_file;
std::string save_vocab_file, read_vocab_file;
struct word *vocab;
std::ifstream fin;

int window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

size_t vocab_size = 0, layer1_size = 100,  layer1_size_aligned;;
long long train_words = 0, word_count_actual = 0, file_size = 0;
int epochs = 5;
double alpha = 0.025, starting_alpha, sample = 1e-3;
double *syn0;
int * sen;
clock_t start;
size_t table_size = 100;
int *table;



// calculate wj^0.75 / SUM i= 0->n (w_i^0.75)
void initUnigramDistribuiton() {
    table = new int[table_size];
    double a = 0.75;
    double sum_p = 0.0, p_w;
    size_t index = 0, prev_index = 0;
    for (size_t i = 0; i < vocab_list.size(); i++) sum_p += pow(vocab_list[i].count, a);
    for (size_t i = 0; i < vocab_list.size(); i++) {
        p_w += pow(vocab_list[i].count, a) / sum_p;
        prev_index = index;
        index = p_w * table_size;
        std::cout << "Word: " << vocab_list[i].word << " Prob: " << p_w << " Index: " << index << std::endl;
        for (size_t j = prev_index; j < index && j < table_size; j++) {
            table[j] = i;
        }
    }
    while(index < table_size) table[index++] = vocab_list.size() - 1;
}


void destroy() {
    delete[] table;
}



void testvocab() {
    train_corpus_file = "corpus/minimal.txt";
    save_vocab_file = "vocab/minimal_vocab.txt";
    loadFromVocabFile(train_corpus_file);
    saveVocab();
}

void testtable() {
    initUnigramDistribuiton();
    std::map<int, int> count_map;
    for (size_t i = 0; i < table_size; i++) {
        std::cout << "Table[" << i << "] = " << table[i] << std::endl;
        count_map[table[i]]++;
    }
    for (const auto& pair : count_map) {
        std::cout << "Word index: " << pair.first << " Count in table: " << pair.second << std::endl;
    }
}

int main() {
    std::cout << "testing voacab" << std::endl;
    testvocab();
    std::cout << "testing table" << std::endl;
    testtable();

    destroy();
    return 0;
}