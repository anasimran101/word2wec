#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>


#include "vocab.h"
 

int loadWordFromFile(std::string& word, std::ifstream& file) {
    int a = 0;
    char ch;
    word.clear();
    while (!file.eof()) {
        ch = file.get();
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') file.unget();
                break;
            }
            if (ch == '\n') {
                word = "</s>";
                return 0;
            } else continue;
        }
        word.push_back((char)ch);
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word.push_back(0);
    return 0;
}

int getWordIndex(const std::string& word) {
    auto it = vocab_hash.find(word);
    if (it != vocab_hash.end()) {
        return it->second;
    } else {
        return -1;
    }
    return -1;
}


// load from file with { word count }format
int loadFromVocabFile(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Error opening vocabulary file: " << vocab_file << std::endl;
        return -1;
    }
    return 0;
}

int loadFromTrainFile(const std::string& vocab_file) {

}
int saveVocab();