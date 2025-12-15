#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>


#define MAX_STRING 100

struct word {
    std::string text;
    int count;
};

std::unordered_map<std::string, int> vocab_hash;
std::vector<word> vocab_list;

int loadWordFromFile(std::string& word, std::ifstream& file);
int loadFromVocabFile(const std::string& vocab_file);
int loadFromTrainFile(const std::string& vocab_file);
int saveVocab();