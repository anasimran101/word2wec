#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vocab.h>
#include <vector>
#include <unordered_map>


std::unordered_map<std::string, int> vocab_hash;
std::vector<word> vocab_list;

std::string train_corpus_file, output_file;
std::string save_vocab_file, read_vocab_file;
struct word *vocab;
std::ifstream fin;

int window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

int vocab_size = 0, layer1_size = 100,  layer1_size_aligned;;
long long train_words = 0, word_count_actual = 0, file_size = 0;
int epochs = 5;
double alpha = 0.025, starting_alpha, sample = 1e-3;
double *syn0;
int * sen;
clock_t start;



void testvocab() {
    train_corpus_file = "corpus/minimal.txt";
    save_vocab_file = "vocab/minimal_vocab.txt";
    loadFromVocabFile(train_corpus_file);
    saveVocab();
}

int main() {
    std::cout << "testing voacab" << std::endl;
    testvocab();
    return 0;
}