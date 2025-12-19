#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>


#include "vocab.h"
 

int loadWordFromFile(std::string& word, std::ifstream& file) {
    char ch;
    word.clear();
    while (file.get(ch)) {
        if (ch == '\r') continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (!word.empty()) {
                if (ch == '\n') file.unget();
                break;
            }
            if (ch == '\n') {
                word = "</s>";
                break;
            } else continue;
        }
        word.push_back(ch);
    }
    return !word.empty();
}

int getWordIndex(const std::string& word) {
    auto it = vocab_hash.find(word);
    if (it != vocab_hash.end()) {
        return it->second;
    }
    return -1;
}

int insertWord(const word& word) {
    int i = getWordIndex(word.word);
    if (i != -1) {
        return i;
    }
    int index = vocab_list.size();
    vocab_list.push_back(word);
    vocab_hash[word.word] = index;
    return index;
}




int loadFromTrainFile(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Error opening vocabulary file: " << vocab_file << std::endl;
        return -1;
    }
    insertWord({"</s>", 0}); // add sentence end token as first word
    std::string word;
    int i;
    while(loadWordFromFile(word, file)){
        
        //std::cout << "Loaded word: " << word << std::endl;
        i = getWordIndex(word);
        if(i != -1) {
            vocab_list[i].count += 1;
            continue;
        }
        insertWord({word, 1});
    }
    vocab_size = vocab_list.size();
    file.close();
    return 0;
}

// load from file with { word count }format
int loadFromVocabFile(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Error opening training file: " << vocab_file << std::endl;
        return -1;
    }
    insertWord({"</s>", 0}); // add sentence end token as first word
    std::string word;
    int count = 0;
    while (!file.eof()){
        file >> word >> count;
        insertWord({word, count});
    }
    vocab_size = vocab_list.size();
    file.close();
    return 0;
}
int saveVocab() {

    std::ofstream file(save_vocab_file, std::ios::out | std::ios::trunc);
    if(!file.is_open()) {
        std::cerr << "Error opening file to save vocabulary: " << save_vocab_file << std::endl;
        return -1;
    }
    for (auto & e: vocab_list)
    {
        file << e.word << " " << e.count << "\n";
    }
    file.close();
    return 0;
}