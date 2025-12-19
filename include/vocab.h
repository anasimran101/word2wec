#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>


#define MAX_STRING 100

struct word {
    std::string word;
    int count;
};

typedef struct word word;

extern std::unordered_map<std::string, int> vocab_hash;
extern std::vector<word> vocab_list;
extern std::string train_corpus_file, output_file;
extern std::string save_vocab_file, read_vocab_file;
extern size_t vocab_size;
extern int train_words;


int loadWordFromFile(std::string& word, std::ifstream& file);
int loadFromVocabFile(const std::string& vocab_file);
int loadFromTrainFile(const std::string& train_corpus_file);
int getWordIndex(const std::string& word);
int saveVocab();
